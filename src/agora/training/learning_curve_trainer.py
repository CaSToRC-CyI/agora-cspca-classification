import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from agora.training.base_config import BaseConfig
from agora.training.base_trainer import BaseTrainer, seed_worker, set_global_seeds


class MutableSubset(Dataset):
    def __init__(self, subset):
        """Initialize a mutable subset for the training dataset.

        Args:
            subset: Training subset generated from stratified split
        """
        self.dataset = subset.dataset
        # self.indices = subset.indices
        self.indices = np.array(subset.indices, dtype=np.int64)
        # Copy labels for this subset (NumPy array or torch tensor)
        self.labels = self.dataset.cached_labels[self.indices].copy()

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        return self.dataset[base_idx]  # pass through dict as-is

    def __len__(self):
        return len(self.indices)


class LearningCurveTrainer(BaseTrainer):
    """Subclass of BaseTrainer for running learning curve experiments.

    Stores the original full train dataset after setup_data() so it can be
    repeatedly subsampled without re-reading from disk. Val/test sets are
    never touched.
    """

    @classmethod
    def build_for_dry_run(cls, cfg: BaseConfig):
        instance = cls.__new__(cls)  # skip __init__
        instance.cfg = cfg
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.use_clinical = cfg.model.use_clinical
        instance.att_fusion = cfg.model.att_fusion
        instance.rep_seed = cfg.training.seed
        set_global_seeds(instance.rep_seed)
        instance.setup_data()
        instance._full_train_dataset = instance.train_loader.dataset
        return instance

    def __init__(self, cfg: BaseConfig):
        super().__init__(cfg)
        # Snapshot the full train dataset after BaseTrainer sets it up.
        # resplit_data() re-snapshots this for every learning-curve repeat.
        self._full_train_dataset = self.train_loader.dataset

    def resplit_data(self, seed: int) -> None:
        """Regenerate the train/val/test split with a new seed.

        Called once per learning-curve repeat so that each repeat trains and
        evaluates on an independent stratified split (ShuffleSplit-style),
        mirroring scikit-learn's cross-validated learning curve where the
        variance across repeats comes from re-splitting the data — not just
        from re-initialising the model. Rebuilds val/test loaders to match the
        new split and re-snapshots the full train dataset so update_fraction
        subsamples from this repeat's training pool.

        After calling this, call update_fraction + reset_train_loader to set
        the training subset, then reset_run to rebuild the model.
        """
        self.rep_seed = seed
        set_global_seeds(seed)
        # Rebuilds self.train_loader / self.val_loader / self.test_loader and
        # re-derives self.pos_weight and self.clinical_stats from the new split.
        self.setup_data()
        self._full_train_dataset = self.train_loader.dataset

    def update_fraction(self, fraction: float, seed: int) -> MutableSubset:
        """Return a stratified subsample of the full train dataset.

        Args:
            fraction: Proportion of training data to use (0, 1].
            seed: Random seed for reproducibility across repeats.

        Returns:
            A new Subset with ~fraction * len(full_train) samples,
            preserving class balance via stratified sampling.
        """
        if not (0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        rng = np.random.default_rng(seed)
        randomized = MutableSubset(self._full_train_dataset)

        if fraction == 1.0:
            return randomized

        neg_ind = np.where(randomized.labels == 0)[0]
        pos_ind = np.where(randomized.labels == 1)[0]

        n_neg = max(1, round(len(neg_ind) * fraction))
        n_pos = max(1, round(len(pos_ind) * fraction))

        # Shuffle each class ONCE per seed, then take prefixes so that smaller
        # fractions are nested inside larger ones (size-0.1 ⊂ size-0.33 ⊂ ...).
        # Because the permutation is fully determined by `seed` and the (fixed)
        # class-index arrays, every fraction in a repeat sees the same ordering,
        # so growing the fraction only adds samples — it never swaps to a
        # different random composition. This isolates the data-size effect
        # within a repeat (matches scikit-learn's train[:n_train_samples]),
        # while keeping the subset stratified across the two classes.
        neg_perm = rng.permutation(neg_ind)
        pos_perm = rng.permutation(pos_ind)
        chosen_neg = neg_perm[:n_neg]
        chosen_pos = pos_perm[:n_pos]

        chosen_relative = np.concatenate([chosen_neg, chosen_pos])
        rng.shuffle(chosen_relative)

        randomized.indices = randomized.indices[chosen_relative]
        randomized.labels = randomized.labels[chosen_relative]

        return randomized

    def reset_train_loader(self, subset: MutableSubset) -> None:
        """Replace self.train_loader with a new DataLoader over the given subset.

        Weighted sampling is re-computed for the new subset if enabled in cfg.
        """
        sampler, shuffle = None, True
        if self.cfg.training.weighted_sampling:
            sampler = self._get_weighted_sampler(subset)
            shuffle = False

        g = torch.Generator()  # pyright: ignore[reportPrivateImportUsage]
        g.manual_seed(self.rep_seed)

        self.train_loader = DataLoader(
            subset,
            batch_size=self.cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.training.num_workers,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def reset_run(self, seed: int) -> None:
        """Reset training state for a fresh run.

        Note: _full_train_dataset is intentionally NOT reset — the original
        train Subset is preserved across all fraction runs to avoid re-reading
        from disk. Call update_fraction + reset_train_loader separately to
        control what data the next run trains on.
        """
        set_global_seeds(seed)
        self.rep_seed = seed

        # Fresh directories (new timestamp + uuid per run)
        self.setup_directories()

        # Rebuild model from scratch
        self._build_model()
        self._set_loss_fn()
        self._set_optimizer()

        # No checkpoint to load between LC runs
        self.start_epoch = 0

        self._set_scheduler()

        # Reset metric tracking state
        self.best_threshold = float("-inf")
        self.best_threshold_tgt_recall = float("inf")
        self.best_metric = float("-inf")
        self.best_loss = float("inf")
        self.no_improvement_count = 0
        self.best_ckpt_path = None

        # Reset AMP scaler
        self.scaler = torch.amp.GradScaler(device=self.device)

        self.save_params_json(self.cfg)
