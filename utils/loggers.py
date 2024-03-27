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

        shutil.copyfile(video_path, destination_file_path)

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


class LoggingManager:
    def __init__(self, loggers=None):
        """
        Initializes the LoggingManager with a list of loggers.

        Parameters:
        - loggers: A list of logger instances that inherit from BaseLogger.
        """
        self.loggers = loggers if loggers is not None else []

    def add_logger(self, logger):
        """
        Adds a logger to the logging manager.

        Parameters:
        - logger: An instance of a logger that should be added to the manager.
        """
        self.loggers.append(logger)

    def log_scalar(self, tag, value, step=None):
        """
        Logs a scalar value across all loggers.

        Parameters are passed directly to the logger's log_scalar method.
        """
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        """
        Logs a histogram across all loggers.
        """
        for logger in self.loggers:
            logger.log_histogram(tag, values, step)

    def log_figure(self, tag, figure, step):
        """
        Logs a figure across all loggers.
        """
        for logger in self.loggers:
            logger.log_figure(tag, figure, step)

    def log_video(self, tag, video_path, step=None, fps=20):
        """
        Logs a video across all loggers.
        """
        for logger in self.loggers:
            logger.log_video(tag, video_path, step, fps)

    def log_hparams(self, hparams, loss):
        """
        Logs hyperparameters across all loggers.
        """
        for logger in self.loggers:
            logger.log_hparams(hparams, loss)

    def log_text(self, tag, text):
        """
        Logs text across all loggers.
        """
        for logger in self.loggers:
            logger.log_text(tag, text)

    def log_model(self, model):
        for logger in self.loggers:
            logger.log_model(model)

    def finish(self):
        """
        Calls the finish method on all loggers to properly close them before the program terminates.
        """
        for logger in self.loggers:
            logger.finish()
