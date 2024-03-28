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

    def log_text(self, tag, text):
        raise NotImplementedError

    def log_model(self, model):
        raise NotImplementedError

    def log_path(self, tag, path, type=None):
        raise NotImplementedError

    def log_args(self, args):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class TensorBoardLogger(BaseLogger):
    DEFAULT_LAYOUT = {
        "Losses": {
            "Last Losses": ["Multiline", ["Loss/last_both", "Loss/last_pos", "Loss/last_vel"]],
            "Average Losses": ["Multiline", ["Loss/avg_both", "Loss/avg_pos", "Loss/avg_vel"]],
            "% Losses": ["Multiline",
                         ["Loss/perc_pos", "Loss/perc_vel", "Loss/perc_pos_vs_vel_l1", "Loss/perc_pos_vs_vel_l2"]],
        },
    }

    def __init__(self, writer, layout=DEFAULT_LAYOUT):
        self.writer = writer

        if layout is not None:
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
        destination_file_path = os.path.join(self.get_logdir(), f"{tag.replace('/', '_')}{ext}")
        self.log_path(None, destination_file_path)

    def log_hparams(self, hparams, loss):
        self.writer.add_hparams(hparams, {'loss': loss})
        # for key, value in hparams.items():
        #     if isinstance(value, (int, float)):
        #         self.writer.add_scalar(key, value)

    def log_text(self, tag, text):
        self.writer.add_text(tag, text)

    def log_model(self, model):
        import torch
        torch.save(model, os.path.join(self.writer.get_logdir(), "model.pth"))
        torch.save(model.state_dict(), os.path.join(self.writer.get_logdir(), "model_state_dict.pth"))

    def log_path(self, tag, path, type=None):
        destination_path = os.path.join(self.get_logdir(), os.path.basename(path))

        if os.path.isfile(path):
            shutil.copyfile(path, destination_path)
        elif os.path.isdir(path):
            shutil.copytree(path, destination_path)

    def log_args(self, args):
        import json
        args_dict = vars(args) if not isinstance(args, dict) else args
        self.log_text('args', ', '.join(f'{k}={v}' for k, v in args_dict.items()))
        with open(os.path.join(self.get_logdir(), 'training_args.json'), 'w') as f:
            json.dump({"args": args_dict}, f, indent=4)

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

    def log_video(self, tag, path, step=None, fps=20):
        self.wandb_run.log({tag: wandb.Video(path, fps=fps, format="mp4"), 'epoch': step})

    def log_hparams(self, hparams, loss):
        # already implemented in config on init
        pass

    def finish(self):
        self.wandb_run.finish()

    def log_text(self, tag, text):
        self.wandb_run.log({tag: text})

    def log_model(self, model):
        import torch
        import tempfile

        # Create temporary files for the model and its state_dict
        with tempfile.NamedTemporaryFile(delete=False) as tmp_model_file, tempfile.NamedTemporaryFile(
                delete=False) as tmp_state_dict_file:
            # Save the model and its state_dict to these temporary files
            torch.save(model, tmp_model_file.name)
            torch.save(model.state_dict(), tmp_state_dict_file.name)

            # Create a wandb Artifact for the model
            artifact = wandb.Artifact("model_artifact", type="model")
            # Add the temporary files to the artifact
            artifact.add_file(tmp_model_file.name, "model.pth")
            artifact.add_file(tmp_state_dict_file.name, "model_state_dict.pth")

            # Log the artifact to wandb
            wandb.log_artifact(artifact)

    def log_path(self, tag, path, type=None):
        if type == "dataset":
            print("Logging datasets is disabled in wandb. If you want to manually disable this check change the type.")
            return

        artifact = None
        if os.path.isfile(path):
            artifact = wandb.Artifact(name=tag, type=type)
            artifact.add_file(path)
        elif os.path.isdir(path):
            artifact = wandb.Artifact(name=tag, type=type)
            artifact.add_dir(path)

        self.wandb_run.log_artifact(artifact)

    def log_args(self, args):
        import json
        import tempfile
        import os

        args_dict = vars(args) if not isinstance(args, dict) else args
        self.wandb_run.config.update(args_dict, allow_val_change=True)

        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as tmp:
            json.dump({"args": args_dict}, tmp, indent=4)
            tmp_path = tmp.name  # Save the path to delete the file later

        # Use the temporary file path after closing the file
        self.log_path("training_args", tmp_path, "args")

        # Manually delete the temporary file
        os.remove(tmp_path)


class LoggingManager:
    def __init__(self, loggers=None):
        self.loggers = loggers if loggers is not None else []

    def add_logger(self, logger):
        self.loggers.append(logger)

    def log_scalar(self, tag, value, step=None):
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        for logger in self.loggers:
            logger.log_histogram(tag, values, step)

    def log_figure(self, tag, figure, step):
        for logger in self.loggers:
            logger.log_figure(tag, figure, step)

    def log_video(self, tag, video_path, step=None, fps=20):
        for logger in self.loggers:
            logger.log_video(tag, video_path, step, fps)

    def log_hparams(self, hparams, loss):
        for logger in self.loggers:
            logger.log_hparams(hparams, loss)

    def log_text(self, tag, text):
        for logger in self.loggers:
            logger.log_text(tag, text)

    def log_model(self, model):
        for logger in self.loggers:
            logger.log_model(model)

    def log_path(self, tag, path, type=None):
        for logger in self.loggers:
            logger.log_path(tag, path, type)

    def log_args(self, args):
        for logger in self.loggers:
            logger.log_args(args)

    def finish(self):
        for logger in self.loggers:
            logger.finish()
