import os
from dadp.models.dynamics_network import VAE
import gym
import dadp.sampling as sampling
import dadp.utils as utils
import torch
import logging
import gc
import psutil, os

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'
args = Parser().parse_args('plan')

env = gym.make(args.dataset)
model = VAE(env.observation_space.shape[0] + env.observation_space.shape[0],
            env.action_space.shape[0], args.device).to(args.device)

model.load_state_dict(torch.load(args.model_path, map_location=args.device))
print("model_path:",args.model_path)
model.eval()

parent_directory = os.path.dirname(args.savepath)
args.savepath = os.path.join(
    parent_directory,
    f't_start{args.t_start}_alpha{args.alpha}_scale{args.scale}_scaleDev{args.scale_deviation}'
)
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

fullpath = os.path.join(args.savepath, 'args.json')
args.save()

# --------------------------------- setup ---------------------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, device= args.device
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, device= args.device
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
## initialize value guide
value_function = value_experiment.ema
# guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide_config = utils.Config(args.guide, model=value_function, dynamics_model=model, env=env, verbose=False)
guide = guide_config()
## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    scale_deviation=args.scale_deviation,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_deviation_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    t_start=args.t_start,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)
policy = policy_config()

n = args.num_plans
for i in range(args.plan_start,n):
    utils.set_seed(i)
    plan_directory = os.path.join(args.savepath, f"plan_{i}")
    os.makedirs(plan_directory, exist_ok=True)

    logger_config = utils.Config(
        utils.Logger,
        renderer=renderer,
        logpath=plan_directory,
        vis_freq=args.vis_freq,
        max_render=args.max_render,
    )

    logger = logger_config()

    # --------------------------------- plan ---------------------------------#

    observation = env.reset()
    ## observations for rendering
    rollout = [observation.copy()]
    total_reward = 0

    for t in range(args.max_episode_length):
        if t % 50 == 0:
            print(plan_directory, flush=True)

        state = env.state_vector().copy()
        conditions = {0: observation}

        action, samples = policy(conditions, batch_size=args.batch_size)
        next_observation, reward, terminal, _ = env.step(action)

        total_reward += reward
        score = env.get_normalized_score(total_reward)

        if(t % 20 == 0):
            print(
                f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                f'values: {samples.values[:5]}',
                flush=True,
            )

        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        logger.log(t, samples, state, rollout)
        if terminal:
            break

        observation = next_observation

    logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
    del logger

    gc.collect()

env.close()

