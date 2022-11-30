from random import choices, seed, uniform
from torchvision import datasets, transforms
import os
import torch
import numpy as np
import pandas as pd

from .image_utils import (
    convert_image_from_gray_to_color_by_idx,
    threshold_grayscale_image,
    convert_image_from_gray_to_color_by_name,
    add_sp_noise,
)


def create_record_from_im_and_label(im, label):
    img = threshold_grayscale_image(im)
    digit = get_digit(label)
    parity = get_parity(label)
    magnitude = get_magnitude(label)
    curviness = get_curviness(label)
    return img, label, digit, parity, magnitude, curviness


def create_mnist_tables():
    """
    Creates dataframe of the MNIST dataset in the format that they need to be in to start the
    workshop. In particular, images are thresholded, and each img corresponds to a single row
    with the following data:
        ['img', 'label', 'digit', 'parity', 'magnitude']
    where digit, parity, and magnitude are the groups we will assess bias on.
    """
    root = "data"

    train_mnist = datasets.mnist.MNIST(
        root, train=True, download=True, transform=None, target_transform=None
    )
    test_mnist = datasets.mnist.MNIST(
        root, train=False, download=True, transform=None, target_transform=None
    )

    train_df = pd.DataFrame.from_records(
        (create_record_from_im_and_label(im, label) for (im, label) in train_mnist),
        columns=["img", "label", "digit", "parity", "magnitude", "curviness"],
    )

    test_df = pd.DataFrame.from_records(
        (create_record_from_im_and_label(im, label) for (im, label) in test_mnist),
        columns=["img", "label", "digit", "parity", "magnitude", "curviness"],
    )

    return train_df, test_df


def create_colored_mnist_tables(p):
    """
    Creates dataframe of the MNIST dataset in the format that they need to be in to start the
    workshop, with specific color augmentations parameterized by input float (p). Colors are
    determined via the `color_data_stochastically_by_label` helper function which documents the
    specific methodology for coloring.

    In particular, images are thresholded, and each img corresponds to a single row with the
    following data:
        ['img', 'label', 'digit', 'parity', 'magnitude', 'color']

    p: float
        parameter that defines coloring behavior
    """
    train_df, test_df = create_mnist_tables()
    train_df = _color_data_stochastically_by_label(train_df, "img", "label", p)
    test_df = _color_data_stochastically_by_label(test_df, "img", "label", p)
    return train_df, test_df


def _color_data_stochastically_by_label(df, img_col, label_col, p, random_seed=0):
    """
    Takes in a dataframe of images with labels 0-9 and for each image recolors it to have a black
    background with the digit d colored with the dth color with probability p and the d+1
    (mod 10)th color with probability 1-p.
    """
    seed(random_seed)
    colored_df = df.copy()
    colored_df.is_copy = None
    colors = []
    for i, (img, digit) in enumerate(zip(df[img_col], df[label_col])):
        color_idx = (digit + (uniform(0, 1) > p)) % 10
        colored_img = convert_image_from_gray_to_color_by_idx(img, color_idx)
        colored_df.at[i, img_col] = colored_img
        colors.append(color_idx)

    # Add the color column, since we went in order this should work fine
    colored_df["color"] = colors
    return colored_df


def bicolor_data_stocastically_by_label(
    df, img_col, label_col, p, color_a="orange", color_b="blue", random_seed=0
):
    """
    Create a copy of the DataFrame df where the entire set is colored on a two-color scheme, using
    the input float p.

    Args:
        df: DataFrame with 'img' and 'label' columns
        p: float describing the proportion of images that will be colored red in the output
            DataFrame.
    Returns:
        c_df: df augmented with color in proportion p
    """
    seed(random_seed)
    colored_df = df.copy()
    colored_df.is_copy = None
    colors = []

    for i, (img, digit) in enumerate(zip(df[img_col], df[label_col])):
        # Decide whether to use color A or B, based on the param p
        clr = color_a if (uniform(0, 1) > p) else color_b
        colored_img = convert_image_from_gray_to_color_by_name(img, clr)
        colored_df.at[i, img_col] = colored_img
        colors.append(clr)

    # Add the color column, since we went in order this should work fine
    colored_df["color"] = colors
    return colored_df


def create_color_biased_mnist_tables(color_probs_train, color_probs_test):
    """
    Creating a dataset meant to have bias on a particular digit.
    """
    train_df, test_df = create_mnist_tables()
    train_df = color_mnist_data_with_dict_per_digit(train_df, color_probs_train)
    test_df = color_mnist_data_with_dict_per_digit(test_df, color_probs_test)
    return train_df, test_df


def color_mnist_data_with_dict_per_digit(df, color_probs, random_seed=0):
    """
    Takes in a dict of up to 10 entries and for each digit d color_probs[d].
    color_probs: dict[int, dict[string, float]]
        A dict with up to 10 keys being a subset of the digits 0-9
        color_probs[d] is a dict that defines the color distribution of digit d in the dataset.
        For instance, if we 3s to be 40% red and 60% green, then color_probs[3] should be
        `{'red': 0.4, 'green':0.6}`. The sum of values of each dictionary must be 1.0 or a
        ValueError will be raised. The valid colors are 'red', 'green', 'blue', 'yellow', 'cyan',
        'magenta'.
    """
    seed(random_seed)
    digit_dfs = []

    # For each unique digit d create a sub dataframe
    for d in range(10):
        # Only include the digit if we have a color distribution defined for it
        if d not in color_probs:
            continue
        color_dist = color_probs[d]

        # Ensure the sum of the color dist is 1.0
        try:
            assert sum(color_dist.values()) == 1.0
        except Exception as e:
            raise ValueError(
                f"The sum of color_probs[d].values() must be 1.0 but for digit {d} it was "
                f"{color_dist.values()}"
            )

        # Copy the subset of the data corresponding to this digit
        digit_df = df[df["label"] == d].copy()

        # Determining the sampling list and weights for each instance (NOTE: this is random/noisy
        # sampling)
        color_list, color_weights = zip(*color_dist.items())

        # Choose the colors for each image
        rand_colors = choices(color_list, k=len(digit_df), weights=color_weights)
        colors = []
        records = []

        # Iterate over each row of the dataset and color accordingly
        for i, record in enumerate(digit_df.itertuples()):
            color = rand_colors[i]
            img = record.img
            colored_img = convert_image_from_gray_to_color_by_name(img, color)
            colors.append(color)
            records.append(
                (
                    colored_img,
                    record.label,
                    record.digit,
                    record.parity,
                    record.magnitude,
                    record.curviness,
                    color,
                )
            )

        # Add the color column and add this df to our list
        new_df = pd.DataFrame.from_records(
            records,
            columns=["img", "label", "digit", "parity", "magnitude", "curviness", "color"],
        )
        digit_dfs.append(new_df)

    # Concatenate all the individual digit dfs
    return pd.concat(digit_dfs)


def resample_dataset(
    df, group_col, group_props, group_name_to_index_dict, random_seed=0
):
    """
    Returns a new dataframe such that for each unique subgroup in group column (which should be
    indexed 0, 1, n_grps-1) the resulting dataframe only has group_prop[i] proportion of rows
    (randomly selected) for members of group i.

    For instance `subsample_dataframe_by_group(df, 'digit', [1., 1., 1., 1., 1., 1., 0.5, 1., 1.,
    1.])` would correspond to a dataset in which only half the original rows labeled 6 remain in
    the dataset.
    """
    group_sizes = df[group_col].value_counts()
    group_weights = np.array(group_props) / max(group_props)  # N

    # Multiply the proportions (normalized so max is 1.0) by min size to guarentee we have enough
    # instances and retain proportions of the data
    subsampled_sizes = group_weights * min(group_sizes)

    # Concatenate the samples from each group individually
    random_state = np.random.RandomState(random_seed)

    return pd.concat(
        dff.sample(
            n=int(subsampled_sizes[group_name_to_index_dict[i]]),
            random_state=random_state,
        )
        for i, dff in df.groupby(group_col)
    )


def get_digit(label):
    """
    Returns the string of the digit group for the label.
    """
    return str(label)


def get_parity(label):
    """
    Returns the string of the parity group for the label.
    """
    return "odd" if label % 2 else "even"


def get_magnitude(label):
    """
    Returns the string of the magnitude group for the label.
    """
    magnitude_dict = {
        0: "small",
        1: "small",
        2: "small",
        3: "small",
        4: "medium",
        5: "medium",
        6: "medium",
        7: "large",
        8: "large",
        9: "large",
    }
    return magnitude_dict[label]


def get_curviness(label):
    """
    Returns the string of the curviness group for the label.
    """
    curves = [3, 8, 0]
    lines = [1, 4, 7]
    if label in curves:
        return "curvy"
    elif label in lines:
        return "linear"
    else:
        return "neutral"


def create_starting_data():
    """
    Download MNIST dataset and resample, so that the workshop can start from the biased version and
    address rebalancing after the initial bias analysis.

    Both train_data and test_data have been augmented with parity, magnitude, curviness, and color.

    train_data:
        Distribution of identities is resampled to 0.3 3's and 0.3 8's. This will affect accuracy
        on identity (3's 8's perform worse), on curviness ('curvy' performs worse than 'linear'). Color
        distribution will be .8 red / .2 green for all number identities, which may affect accuracy
        on color.

    test_data:
        Distribution of identities ('label') is uniform color distribution is .5 red / .5 green for
        all number identities, with the exception of 6 which has color distribution .2 red / .8
        green, which may affect accuracy on 6's.

    Return train_data and test_data
    """
    # Download mnist: default distribution (NOTE: this includes parity, magnitude, and curviness)
    train_data, test_data = create_mnist_tables()

    # Add color distribution .8 red / .2 green
    num_colorful_train_data = bicolor_data_stocastically_by_label(
        train_data, "img", "label", 0.8, color_a="orange", color_b="blue"
    )
    num_colorful_test_data = bicolor_data_stocastically_by_label(
        test_data, "img", "label", 0.5, color_a="orange", color_b="blue"
    )

    # Resample so that 3's and 8's are under-represented
    sampled_num_colorful_train_data = resample_dataset(
        num_colorful_train_data,
        group_col="digit",
        group_props=[1, 1, 1, 0.3, 1, 1, 0.3, 1, 1, 1],
        group_name_to_index_dict={str(i): i for i in range(10)},
    )

    return sampled_num_colorful_train_data, num_colorful_test_data


def create_augmented_data():
    """
    Create starting dataset without downsampling.

    Simulates the case where we gathered additional data for under-represented groups.
    """
    # Download mnist: default distribution  NOTE: this includes parity, magnitude, and curviness
    train_data, test_data = create_mnist_tables()

    # Add color distribution .8 red / .2 green
    num_colorful_train_data = bicolor_data_stocastically_by_label(
        train_data, "img", "label", 0.8, color_a="orange", color_b="blue"
    )
    return num_colorful_train_data


def create_augmented_data_2():
    """
    Create starting dataset without downsampling.

    Simulates the case where we gathered additional data for under-represented groups.
    """
    # Download mnist: default distribution  NOTE: this includes parity, magnitude, and curviness
    train_data, test_data = create_mnist_tables()

    # Add color distribution .8 red / .2 green
    num_colorful_train_data = bicolor_data_stocastically_by_label(
        train_data, "img", "label", 0.8, color_a="orange", color_b="blue"
    )

    # Resample so that 3's and 8's are under-represented
    sampled_num_colorful_train_data = resample_dataset(
        num_colorful_train_data,
        group_col="digit",
        group_props=[1, 1, 1, 1, 1, 1, 0.3, 1, 1, 1],
        group_name_to_index_dict={str(i): i for i in range(10)},
    )
    return sampled_num_colorful_train_data


def get_val2(df):
    """
    Create a version of val_data where the 4's have salt-and-peper noise.
    """
    val2_df = df.copy()
    val2_df["img"] = val2_df.apply(
        lambda row: add_sp_noise(row["img"], 0.02)
        if row["digit"] == "4"
        else row["img"],
        axis=1,
    )
    return val2_df


def create_dataloader_from_df(
    df, img_col, label_col, dataset_name, data_shuffle_seed, train=True
):
    """
    Creates a pytorch dataloader from our dataframe.
    """
    dataset = make_dataset_from_df(df, img_col, label_col, dataset_name)
    torch.manual_seed(data_shuffle_seed)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64 if train else 1000, shuffle=train
    )
    return data_loader


def create_dataloader_from_df_with_groups(
    df, img_col, label_col, dataset_name, group_names, data_shuffle_seed, train=False
):
    """
    Creates a pytorch dataloader from our dataframe.
    """
    dataset = make_dataset_from_df_with_groups(
        df, img_col, label_col, dataset_name, group_names
    )
    torch.manual_seed(data_shuffle_seed)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64 if train else 1000, shuffle=True
    )
    return data_loader


def make_dataset_from_df(df, img_col, label_col, dataset_name, root="data"):
    """
    Creates an MNISTDataset object from a dataframe.
    """
    root = os.path.join(root, dataset_name)
    data_tuples = list(zip(df[img_col], df[label_col]))
    return MNISTDataset(root, data_tuples)


def make_dataset_from_df_with_groups(
    df, img_col, label_col, dataset_name, group_names, root="data"
):
    """
    Makes an instance of the MNISTDataset class that also stores group membership values for each
    instance.
    """
    root = os.path.join(root, dataset_name)
    data_tuples = list(
        zip(df[img_col], df[label_col], *(df[group] for group in group_names))
    )
    return MNISTDataset(root, data_tuples, group_names=group_names)


class MNISTDataset(datasets.VisionDataset):
    """
    Dataset object that is spawned from our MNISTWrapper, we will create one of these for train and
    one for test.

    Attributes:
        data_tuples (list[Tuple[img, label, (group, parity, magnitude)]]): The raw data
    """

    def __init__(
        self,
        root,
        data_tuples,
        transform=transforms.ToTensor(),
        target_transform=None,
        group_names=(),
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.data_tuples = data_tuples  # The list of data tuples
        self.group_names = group_names

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            data triple: (image, target, group_id) where target is index of the target class
        """
        if len(self.group_names) > 0:
            # Note that we simply list the group values in indices 2,3,4,... so we need to use the
            # * to pack into a tuple
            img, target, *group_tuple = self.data_tuples[index]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target, tuple(group_tuple)
        img, target = self.data_tuples[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data_tuples)
