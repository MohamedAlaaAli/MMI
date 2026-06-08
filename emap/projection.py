"""The pure projection math.

Model-agnostic: takes a tensor F[i, j] = f(mod_a_i, mod_b_j) and returns the
diagonal of the multimodally-additive projection. Works for any trailing shape
(scalar logits, class-vector logits, per-pixel logit maps, ...) because the
projection decouples across output dimensions (paper Appendix A).
"""
import torch


def project_logits(F: torch.Tensor) -> torch.Tensor:
    """Multimodally-additive projection of an N x N logit grid.

    Args:
        F: tensor of shape [N, N, ...] where F[i, j] = f(mod_a_i, mod_b_j).
           Trailing dims are arbitrary (e.g. [C], [C, H, W], or scalar).

    Returns:
        f_hat diagonal of shape [N, ...] with
            f_hat[i] = row_mean[i] + col_mean[i] - overall_mean
        where row_mean[i] = mean_j F[i, j], col_mean[i] = mean_j F[j, i],
        overall_mean = mean_{i, j} F[i, j].
    """
    return F.mean(dim=1) + F.mean(dim=0) - F.mean(dim=(0, 1))


def appendix_g_test():
    """Reproduce the worked example from Hessel & Lee, Appendix G.

    Expected projected diagonal: [-0.8, 2.1, 0.5].
    """
    F = torch.tensor(
        [[-1.3, 0.3, -0.2],
         [0.8, 3.0, 1.1],
         [1.1, -0.1, 0.7]],
        dtype=torch.float64,
    )
    expected = torch.tensor([-0.8, 2.1, 0.5], dtype=torch.float64)
    got = project_logits(F)
    assert torch.allclose(got, expected, atol=1e-9), f"Appendix G mismatch: got {got.tolist()}"
    return got


if __name__ == "__main__":
    appendix_g_test()
    print("Appendix G projection test passed.")
