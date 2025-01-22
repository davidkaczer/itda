import torch
from sklearn.linear_model import orthogonal_mp


def omp_incremental_cholesky(D, x, n_nonzero_coefs, device=None):
    if device is None:
        device = D.device
    D = D.to(device)
    x = x.to(device)
    batch_size, n_features = x.shape
    n_atoms = D.shape[0]

    indices = torch.zeros(
        (batch_size, n_nonzero_coefs), device=device, dtype=torch.long
    )
    residual = x.clone()
    selected_atoms = torch.zeros(
        (batch_size, n_nonzero_coefs, n_features), device=device, dtype=D.dtype
    )
    available_atoms = torch.ones((batch_size, n_atoms), dtype=torch.bool, device=device)

    L = None
    b = None
    Beta = None

    for k in range(n_nonzero_coefs):
        correlations = torch.matmul(residual, D.T)
        abs_correlations = torch.abs(correlations)
        abs_correlations[~available_atoms] = 0
        idx = torch.argmax(abs_correlations, dim=1)
        indices[:, k] = idx
        available_atoms[torch.arange(batch_size, device=device), idx] = False

        a_k = D[idx, :]
        selected_atoms[:, k, :] = a_k
        a_k_x = torch.sum(a_k * x, dim=-1, keepdim=True)

        if k == 0:
            norm_a_k2 = torch.sum(a_k * a_k, dim=-1, keepdim=True)
            L = torch.sqrt(norm_a_k2).unsqueeze(-1)
            b = a_k_x.unsqueeze(-1)
            Beta = b / (L * L)
        else:
            prev_atoms = selected_atoms[:, :k, :]
            c = torch.sum(prev_atoms * a_k.unsqueeze(1), dim=-1)
            y = torch.linalg.solve_triangular(
                L, c.unsqueeze(-1), upper=False, unitriangular=False
            )
            y_norm2 = torch.sum(y.squeeze(-1) ** 2, dim=-1)

            norm_a_k2 = torch.sum(a_k * a_k, dim=-1)
            diff = norm_a_k2 - y_norm2
            diff = torch.clamp(diff, min=0.0)
            r_kk = torch.sqrt(diff + 1e-12)

            L_new = torch.zeros(
                (batch_size, k + 1, k + 1), dtype=L.dtype, device=device
            )
            L_new[:, :k, :k] = L
            L_new[:, k, :k] = y.squeeze(-1)
            L_new[:, k, k] = r_kk
            L = L_new

            b_new = torch.cat([b, a_k_x.unsqueeze(-1)], dim=1)
            b = b_new
            Beta = torch.cholesky_solve(b, L, upper=False)

        A_cur = selected_atoms[:, : k + 1, :]
        recon = torch.bmm(A_cur.transpose(1, 2), Beta).squeeze(-1)
        residual = x - recon

    Beta_final = Beta.squeeze(-1)
    activations = torch.zeros((batch_size, n_atoms), device=device, dtype=D.dtype)
    activations.scatter_(1, indices, Beta_final)
    return activations, Beta_final, residual, indices, selected_atoms


def svd_min_norm_solution(A, B, rcond=None):
    m, n = A.shape[-2], A.shape[-1]
    U, S, Vh = torch.linalg.svd(A, full_matrices=False, driver="gesvda")
    if rcond is None:
        rcond = torch.finfo(S.dtype).eps * max(m, n)
    max_singular = S.max(dim=-1, keepdim=True).values
    tol = rcond * max_singular
    large_singular = S > tol
    S_inv = torch.zeros_like(S)
    S_inv[large_singular] = 1.0 / S[large_singular]
    S_inv = S_inv.unsqueeze(-1)
    U_T_B = torch.matmul(U.transpose(-2, -1), B)
    S_inv_U_T_B = S_inv * U_T_B
    X = torch.matmul(Vh.transpose(-2, -1), S_inv_U_T_B)
    return X


def omp_pytorch(D, x, n_nonzero_coefs):
    batch_size, n_features = x.shape
    n_atoms = D.shape[0]
    indices = torch.zeros(
        (batch_size, n_nonzero_coefs), device=D.device, dtype=torch.long
    )
    residual = x.clone()
    selected_atoms = torch.zeros(
        (batch_size, n_nonzero_coefs, n_features), device=D.device
    )
    available_atoms = torch.ones(
        (batch_size, n_atoms), dtype=torch.bool, device=D.device
    )
    batch_indices = torch.arange(batch_size, device=D.device)

    for k in range(n_nonzero_coefs):
        correlations = torch.matmul(residual, D.T)
        abs_correlations = torch.abs(correlations)
        abs_correlations[~available_atoms] = 0
        idx = torch.argmax(abs_correlations, dim=1)
        indices[:, k] = idx
        available_atoms[batch_indices, idx] = False
        selected_atoms[:, k, :] = D[idx]
        A = selected_atoms[:, : k + 1, :].transpose(1, 2)
        B = x.unsqueeze(2)
        try:
            coef = svd_min_norm_solution(A, B)
        except RuntimeError as e:
            coef = torch.zeros(batch_size, k + 1, 1, device=D.device)
        coef = coef.squeeze(2)
        invalid_coefs = torch.isnan(coef) | torch.isinf(coef)
        if invalid_coefs.any():
            coef[invalid_coefs] = 0.0
        recon = torch.bmm(A, coef.unsqueeze(2)).squeeze(2)
        residual = x - recon
    activations = torch.zeros((x.size(0), n_atoms), device=x.device)
    activations.scatter_(1, indices, coef)
    return activations


def omp_sklearn(D, x, n_nonzero_coefs):
    """
    D: (n, d) torch.Tensor
    x: (b, d) torch.Tensor
    """
    D_np = D.cpu().numpy()  # shape (n, d)
    x_np = x.cpu().numpy()  # shape (b, d)

    # But we must pass X as shape (d, n) to orthogonal_mp, so we do D_np.T
    # And we pass multiple targets as shape (d, b), so we do x_np.T
    codes_np = orthogonal_mp(D_np.T, x_np.T, n_nonzero_coefs=n_nonzero_coefs)
    # codes_np is now shape (n, b)

    codes_np = codes_np.T  # shape (b, n)
    codes_torch = torch.from_numpy(codes_np).to(D.device, D.dtype)

    return codes_torch


def omp_incremental_cholesky_with_fallback(D, x, n_nonzero_coefs, device=None):
    if device is None:
        device = D.device

    activations, Beta_final, residual, indices, selected_atoms = (
        omp_incremental_cholesky(D, x, n_nonzero_coefs, device=device)
    )

    invalid_mask = torch.isnan(Beta_final) | torch.isinf(Beta_final)
    residual_norm = torch.norm(residual, dim=-1)
    high_error_mask = residual_norm > 1e8
    fallback_mask = invalid_mask.any(dim=-1) | high_error_mask

    if fallback_mask.any():
        fallback_indices = torch.where(fallback_mask)[0]
        D_fallback = D
        x_fallback = x[fallback_indices]
        fallback_acts = omp_pytorch(D_fallback, x_fallback, n_nonzero_coefs)
        activations[fallback_indices] = fallback_acts

    return activations
