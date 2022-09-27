import pytorch_lightning as pl


class AntiKDBackdoor(pl.LightningModule):

    def __init__(self, teacher: dict, students: list[dict],
                 trigger: dict) -> None:
        super().__init__()
        self.save_hyperparameters()

        # It's hard to use automatic optimization for anti-kd-backdoor
        # algorithm. More information could be found in the following document.
        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html
        self.automatic_optimization = False

    def training_step(self, *args, **kwargs):
        """Training step of anti-kd-backdoor.

        The algorithm mainly consists of four substeps:
            1. Train backdoor on teacher.
            2. Train backdoor on students.
            3. Train teacher with clean data.
            4. Train students with knowledge distillation.
        """
        self.manual_backward()

    def configure_optimizers(self):
        return super().configure_optimizers()
