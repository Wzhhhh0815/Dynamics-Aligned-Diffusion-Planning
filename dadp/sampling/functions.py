import torch

from dadp.models.helpers import (
    extract,
    apply_conditioning,
)


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0
        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y

@torch.no_grad()
def n_step_guided_deviation_p_sample(
    model, x, cond, t, guide, alpha=1.0,scale=0.001 ,scale_deviation=0.1,t_start = 0,t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad_value = guide.gradients_values(x, cond, t)

        if t[0].item() <= t_start and t[0].item() > t_stopgrad:
            with torch.enable_grad():
                grad_deviation, loss = guide.gradients_deviation(x, cond, t)
        else:
            grad_deviation = torch.zeros_like(grad_value)
            loss = 0

        if scale_grad_by_std:
            grad_value = model_var * grad_value
            grad_deviation = model_var * grad_deviation

        grad_value[t < t_stopgrad] = 0

        x = x + alpha * (scale * grad_value - scale_deviation * grad_deviation)
        x = apply_conditioning(x, cond, model.action_dim)
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y
