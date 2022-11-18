import abc


class BaseTrainer(abc.ABC):

    def __init__(self, *, epochs: int, save_interval: int,
                 epochs_per_validation: int) -> None:
        self._epochs = epochs
        self._save_interval = save_interval
        self._epochs_per_validation = epochs_per_validation

    def train(self) -> None:
        for ep in range(1, self._epochs + 1):
            self.before_train_epoch()
            self.train_epoch()
            self.after_train_epoch()

            if ep == 1 or ep % self._epochs_per_validation == 0:
                self.validation()

    @abc.abstractmethod
    def before_train_epoch(self) -> None:
        ...

    @abc.abstractmethod
    def train_epoch(self) -> None:
        ...

    @abc.abstractmethod
    def after_train_epoch(self) -> None:
        ...

    @abc.abstractmethod
    def validation(self) -> None:
        ...
