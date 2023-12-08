import wandb
import configparser
import os
import shutil


class BaseLogger:
    def log_scalar(self, tag, value, step=None):
        raise NotImplementedError

    def log_histogram(self, tag, values, step):
        raise NotImplementedError

    def get_logdir(self):
        raise NotImplementedError

    def log_figure(self, tag, figure, step):
        raise NotImplementedError

    def log_video(self, tag, video_path, step=None, fps=20):
        raise NotImplementedError

    def log_hparams(self, hparams, loss):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class TensorBoardLogger(BaseLogger):
    def __init__(self, writer):
        self.writer = writer
        layout = {
            "Losses": {
                "Last Losses": ["Multiline", ["Loss/last_both", "Loss/last_pos", "Loss/last_vel"]],
                "Average Losses": ["Multiline", ["Loss/avg_both", "Loss/avg_pos", "Loss/avg_vel"]],
                "% Losses": ["Multiline",
                             ["Loss/perc_pos", "Loss/perc_vel", "Loss/perc_pos_vs_vel_l1", "Loss/perc_pos_vs_vel_l2"]],
            },
        }
        self.writer.add_custom_scalars(layout)

    def log_scalar(self, tag, value, step=None):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def get_logdir(self):
        return self.writer.get_logdir()

    def log_figure(self, tag, figure, step):
        self.writer.add_figure(tag, figure, global_step=step)

    def log_video(self, tag, video_path, step=None, fps=20):
        _, ext = os.path.splitext(video_path)
        destination_file_path = os.path.join(self.get_logdir(), f"{tag.replace('/','_')}{ext}")

        shutil.copyfile(video_path, destination_file_path)

    def log_hparams(self, hparams, loss):
        self.writer.add_hparams(hparams, {'loss': loss})
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value)

    def finish(self):
        pass


class WandBLogger(BaseLogger):
    @staticmethod
    def get_api_key(config_file='config.ini'):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config.get('wandb', 'api_key', fallback=None)

    def __init__(self, project_name, run_name, config=None, config_file='config.ini'):
        apikey = self.get_api_key(config_file)
        if apikey is None:
            raise ValueError("API key not found in configuration.")

        self.api_key = apikey

        wandb.login(key=apikey)
        self.wandb_run = wandb.init(project=project_name, config=config, name=run_name)

    def log_scalar(self, tag, value, step=None):
        if step is not None:
            self.wandb_run.log({tag: value, 'epoch': step})
        else:
            self.wandb_run.log({tag: value})

    def log_histogram(self, tag, values, step):
        self.wandb_run.log({tag: wandb.Histogram(values.detach().cpu()), 'epoch': step})

    def get_logdir(self):
        return None

    def log_figure(self, tag, figure, step):
        self.wandb_run.log({tag: [wandb.Image(figure, caption=tag)], 'epoch': step})

    def log_video(self, tag, video_path, step=None, fps=20):
        self.wandb_run.log({tag: wandb.Video(video_path, fps=fps, format="mp4"), 'epoch': step})

    def log_hparams(self, hparams, loss):
        # already implemented in config on init
        pass

    def finish(self):
        self.wandb_run.finish()
