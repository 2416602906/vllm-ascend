import torch
import torch_npu
import time

# -------------------------------------------------------------------------
# 参数设置
# -------------------------------------------------------------------------
S = 8192  # 测试 8K
H = 32     # Query 头数
KV_H = 32  # KV 头数
D = 128    # 每个头的维度
DTYPE = torch.float16
DEVICE = "npu:0"

# 为了模拟 16K masked (8 tiles) vs (8K masked * 2 + 8K no-mask)
# 我们将 Q, K, V 初始化
q = torch.randn(S, H, D, dtype=DTYPE, device=DEVICE)
k = torch.randn(S, KV_H, D, dtype=DTYPE, device=DEVICE)
v = torch.randn(S, KV_H, D, dtype=DTYPE, device=DEVICE)

# 生成一个简单的 Causal Mask 用于方案 A
# 实际场景中 Mask 可能是从外部传入的 Tensor
mask = torch.triu(torch.ones(S, S, device=DEVICE), diagonal=1).bool()

# -------------------------------------------------------------------------
# 方案 A: 16K 整体计算 (使用硬件内置 Causal 逻辑，不传 Mask Tensor)
# -------------------------------------------------------------------------
def run_approach_a():
    # 当 layout="TND" 时，必须提供 actual_seq_lengths 和 actual_seq_lengths_kv
    actual_seq_lengths = torch.tensor([S], dtype=torch.int32, device=DEVICE)
    actual_seq_lengths_kv = torch.tensor([S], dtype=torch.int32, device=DEVICE)
    
    out, lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        num_heads=H,
        num_key_value_heads=KV_H,
        input_layout="TND",
        atten_mask=None,  # 极致优化：不传 mask tensor，避免 2048 限制
        scale=1.0/8.0,
        sparse_mode=3,  # Causal
        softmax_lse_flag=True,
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv
    )
    return out

# -------------------------------------------------------------------------
# 方案 B: 拆分计算并合并
# 模拟 Zigzag CP 本地块: 2次 8K masked + 1次 8K nomask
# -------------------------------------------------------------------------
def run_approach_b():
    # 模拟 Zigzag CP 布局: Q = [Q_head, Q_tail], K = [K_head, K_tail]
    q_head = q[:S//2]
    q_tail = q[S//2:]
    k_head = k[:S//2]
    k_tail = k[S//2:]
    v_head = v[:S//2]
    v_tail = v[S//2:]
    
    actual_seq_lengths_half = torch.tensor([S//2], dtype=torch.int32, device=DEVICE)
    
    # 1. Q_head vs K_head (Masked/Causal)
    out1, lse1 = torch_npu.npu_fused_infer_attention_score(
        q_head, k_head, v_head,
        num_heads=H, num_key_value_heads=KV_H,
        input_layout="TND", 
        atten_mask=None, # 不传 mask
        scale=1.0/8.0, sparse_mode=3, softmax_lse_flag=True,
        actual_seq_lengths=actual_seq_lengths_half,
        actual_seq_lengths_kv=actual_seq_lengths_half
    )
    
    # 2. Q_tail vs K_head (Nomask/Full)
    out2, lse2 = torch_npu.npu_fused_infer_attention_score(
        q_tail, k_head, v_head,
        num_heads=H, num_key_value_heads=KV_H,
        input_layout="TND", atten_mask=None,
        scale=1.0/8.0, sparse_mode=0, softmax_lse_flag=True,
        actual_seq_lengths=actual_seq_lengths_half,
        actual_seq_lengths_kv=actual_seq_lengths_half
    )

    # 3. Q_tail vs K_tail (Masked/Causal)
    out3, lse3 = torch_npu.npu_fused_infer_attention_score(
        q_tail, k_tail, v_tail,
        num_heads=H, num_key_value_heads=KV_H,
        input_layout="TND", 
        atten_mask=None, # 不传 mask
        scale=1.0/8.0, sparse_mode=3, softmax_lse_flag=True,
        actual_seq_lengths=actual_seq_lengths_half,
        actual_seq_lengths_kv=actual_seq_lengths_half
    )
    
    # 合并逻辑
    # Q_head 对应的结果只需 out1/lse1 (因为它只和 K_head 计算)
    # Q_tail 对应的结果需要合并 out2/lse2 和 out3/lse3
    
    # npu_attention_update 要求 float32
    t_out2 = out2.view(-1, D).to(torch.float32)
    t_lse2 = lse2.view(-1).to(torch.float32)
    t_out3 = out3.view(-1, D).to(torch.float32)
    t_lse3 = lse3.view(-1).to(torch.float32)
    
    merged_out_tail, _ = torch_npu.npu_attention_update(
        [t_lse2, t_lse3],
        [t_out2, t_out3],
        0
    )
    
    # 最终输出拼接触 (Q_head 结果 + 合并后的 Q_tail 结果)
    return torch.cat([out1, merged_out_tail.view(S//2, H, D).to(DTYPE)], dim=0)

# -------------------------------------------------------------------------
# 验证与性能测试
# -------------------------------------------------------------------------
def benchmark(name, func, iterations=100, warmup=20):
    # 预热
    for _ in range(warmup):
        func()
    torch.npu.synchronize()
    
    # 计时
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        func()
    end_event.record()
    
    torch.npu.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / iterations
    print(f"方案 {name} 平均耗时: {elapsed_time:.4f} ms")

if __name__ == "__main__":
    print(f"开始性能对比测试 (S={S}, H={H}, D={D})...")
    benchmark("A (16K Overall Masked)", run_approach_a)
    benchmark("B (Split & Update)", run_approach_b)
