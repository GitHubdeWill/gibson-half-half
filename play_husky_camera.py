from simple_env import HuskyNavigateEnv
from gibson.utils.play import play
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'play_husky_camera.yaml')
print(config_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGB", is_discrete = True, resolution=args.resolution)
    env = HuskyNavigateEnv(config=args.config, gpu_count = 0)
    play(env, zoom=4)
