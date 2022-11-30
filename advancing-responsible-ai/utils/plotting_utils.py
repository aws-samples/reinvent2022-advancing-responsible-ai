from math import sqrt
from random import choice
from string import capwords
from matplotlib import ticker
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from .eval_utils import eval_model_on_groups


def plot_dataset_digits(df, data_col="img", label_col="digit"):
    """
    Plots a sample of digits from the dataframe.
    """
    fig = plt.figure(figsize=(13, 8))
    columns = 6
    rows = 3
    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns * rows):
        # Choose a random index at each stage, so we don't get only the same digit
        idx = choice(range(len(df)))
        img, label = df.iloc[idx][[data_col, label_col]]
        if img.shape[0] == 1:
            pass

        # Create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title("Label: " + str(label))  # set title
        plt.imshow(img)

    plt.show()


def plot_color_for_label(df, dig):
    """
    Plot a histogram of color values for all the images where 'label' == label.
    """
    df.loc[df["label"] == dig].color.value_counts().plot(
        kind="bar", xlabel="Color", ylabel="Count", rot=0
    )


def plot_five(df, data_col="img", label_col="digit", show_label=True):
    """
    Plot the first five entries in df
    """
    fig = plt.figure(figsize=(15, 6))
    columns = 5
    rows = 1
    ax = []  # ax enables access to manipulate each of subplots

    i = 0
    for index, row in df.iterrows():
        img = row["img"]
        if img.shape[0] == 1:
            pass

        # Create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        plt.axis("off")
        plt.imshow(img)
        i += 1

    plt.show()


def plot_one_each(df, data_col="img", label_col="digit", show_label=True):
    """
    Choose one example of each value from label_col and render its data from data_col in the
    DataFrame df.
    """
    fig = plt.figure(figsize=(15, 6))
    columns = 5
    rows = 2
    ax = []  # ax enables access to manipulate each of subplots

    # Keep track of how many items we have plotted
    # Loop through the values for label_col
    for i, label in enumerate(df[label_col].unique()):
        # Grab an item from the subset of the df where label_col==label
        items = df.loc[df[label_col] == label]
        idx = choice(range(len(items)))
        img = items.iloc[idx][data_col]

        if img.shape[0] == 1:
            pass

        # Create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        if show_label:
            ax[-1].set_title("Label: " + str(label))
        plt.axis("off")
        plt.imshow(img)

    plt.show()


def plot_confusion_matrix(df, label_col, pred_col, dataset_name):
    """
    Renders a confusion matrix from a dataframe.
    """
    y_true = df[label_col]
    y_pred = df[pred_col]

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format="g")
    plt.title(f"{dataset_name} Confusion Matrix")
    plt.show()


def plot_confusion_matrices(train_df, test_df, label_col, pred_col, dataset_name):
    """
    Renders a confusion matrix from a dataframe.
    """
    y_true_train = train_df[label_col]
    y_pred_train = train_df[pred_col]

    y_true_test = test_df[label_col]
    y_pred_test = test_df[pred_col]

    cm_train = confusion_matrix(y_true_train, y_pred_train)
    cm_test = confusion_matrix(y_true_test, y_pred_test)

    class_names = y_true_train.unique()

    # Add train to figure
    fig = plt.figure(figsize=(13, 8))
    train_ax = fig.add_subplot(1, 2, 1)
    train_ax.set_title("Train")
    disp_train = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=class_names
    )
    disp_train.plot(cmap=plt.cm.Blues, values_format="g", ax=train_ax)

    # Add test to figure
    test_ax = fig.add_subplot(1, 2, 2)
    test_ax.set_title("Test")
    disp_test = ConfusionMatrixDisplay(
        confusion_matrix=cm_test, display_labels=class_names
    )
    disp_test.plot(cmap=plt.cm.Blues, values_format="g", ax=test_ax)

    plt.suptitle(f"Confusion Matrices for {dataset_name}")
    plt.show()


def plot_confidence_intervals(
    dict_of_dfs,
    group_name,
    dataset_name,
    use_legend=True,
    figsize=(12, 6),
    fontsize=16,
    tick_spacing=1,
    sort_x=False,
    show_error=True,
    marker="o",
):
    """
    Plots confidence intervals from each subgroup given a df with the columns: 'group',
    'correct_count', 'total_count'.

    ci_dict maps a group_name to a tuple ((p, eps), n) whose two values are another value defining
    the CI and the value n denoting the number of instances.

    show_error (bool): If we should show the confidence interval. When false we show point estimate
    only fmt (str):  'o-' or '^-' will determine if the shape is a circle or a triangle
    """
    df = dict_of_dfs[group_name]
    ci_dict = compute_wald_intervals(df)
    n_groups = len(ci_dict)
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
    if show_error:
        title = (
            f"Confidence Intervals on {group_name.capitalize()} Subgroups on {dataset_name}"
            f"Dataset"
        )
    else:
        title = f"Accuracies on {group_name.capitalize()} Subgroups on {dataset_name} Dataset"
    fig.suptitle(title, fontsize=fontsize)

    # Sorted list with entries of the form (group_id, ((p, eps), n)) sorted by values of p
    if sort_x:
        full_data_sorted = sorted(ci_dict.items(), key=lambda x: x[1][0][0])
    else:
        full_data_sorted = list(ci_dict.items())

    # The tick labels are simply the group_ids
    x_ticklabels = [entry[0] for entry in full_data_sorted]
    for idx, (group_id, ((p, eps), n)) in enumerate(full_data_sorted):
        if show_error:
            ax.errorbar(
                y=[p],
                yerr=eps,
                x=[idx],
                markersize=12,
                capsize=12,
                fmt=f"{marker}-",
                label=str(group_id),
            )
        else:
            ax.errorbar(
                y=[p],
                yerr=eps,
                x=[idx],
                markersize=12,
                capsize=12,
                fmt=f"{marker}-",
                label=str(group_id),
                elinewidth=0,
                capthick=0,
            )

    # Only rotate for intersectional groups
    label_rotation = 90 if ("-" in x_ticklabels[0]) else 0
    ax.tick_params(axis="x", labelrotation=90)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xticks(range(n_groups), labels=x_ticklabels)
    if use_legend:
        ax.legend()

    # Make sure values are bounded between 0 and 1
    ax.set_ylim(0.5, 1)

    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


def plot_dual_confidence_intervals(
    dict_of_dfs_1,
    dict_of_dfs_2,
    group_name,
    name_1,
    name_2,
    use_legend=False,
    figsize=(12, 6),
    fontsize=16,
    tick_spacing=1,
    show_error=True,
    marker_1="o",
    marker_2="^",
):
    df_1 = dict_of_dfs_1[group_name]
    df_2 = dict_of_dfs_2[group_name]
    ci_dict_1 = compute_wald_intervals(df_1)
    ci_dict_2 = compute_wald_intervals(df_2)
    n_groups = len(ci_dict_1)

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)

    if show_error:
        title = (
            f"Confidence Intervals on {group_name.capitalize()} Subgroups on {name_1} and "
            f"{name_2}"
        )
    else:
        title = f"Accuracies on {group_name.capitalize()} Subgroups on {name_1} and {name_2}"
    fig.suptitle(title, fontsize=fontsize)

    # Combine the confidence interval dictionaries
    ci_dict = {}
    for k in ci_dict_1:
        ci_dict[k] = (ci_dict_1[k], ci_dict_2[k])
    full_data = list(ci_dict.items())

    # Add all the plot elements
    for idx, (group_id_1, (((p_1, eps_1), n_1), ((p_2, eps_2), n_2))) in enumerate(
        full_data
    ):
        # Add the dataset_1 data in blue
        if show_error:
            plot = ax.errorbar(
                y=[p_1],
                yerr=eps_1,
                x=[idx],
                markersize=12,
                capsize=12,
                fmt=f"{marker_1}-",
                label=str(name_1),
            )
        else:
            plot = ax.errorbar(
                y=[p_1],
                yerr=eps_1,
                x=[idx],
                markersize=12,
                capsize=12,
                fmt=f"{marker_1}-",
                label=str(name_1),
                elinewidth=0,
                capthick=0,
            )

        # Add the dataset_2 data in orange
        color = plot.get_children()[-1].get_color()
        if show_error:
            ax.errorbar(
                y=[p_2],
                yerr=eps_2,
                x=[idx + 0.4],
                markersize=12,
                capsize=12,
                fmt=f"{marker_2}-",
                label=str(name_2),
                color=color,
                alpha=1,
            )
        else:
            ax.errorbar(
                y=[p_2],
                yerr=eps_2,
                x=[idx + 0.4],
                markersize=12,
                capsize=12,
                fmt=f"{marker_2}-",
                label=str(name_2),
                elinewidth=0,
                capthick=0,
                color=color,
                alpha=1,
            )

    # NOTE: The tick labels are simply the group_ids
    x_ticklabels = [entry[0] for entry in full_data]

    # Only rotate for intersectional groups
    label_rotation = 90 if ("-" in x_ticklabels[0]) else 0
    ax.tick_params(axis="x", labelrotation=label_rotation)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xticks(range(n_groups), labels=x_ticklabels)

    # Set up the legend
    if use_legend or True:
        legend_elements = [
            Line2D([0], [0], color="black", marker=marker_1, label=name_1),
            Line2D([0], [0], color="black", marker=marker_2, label=name_2),
        ]
        ax.legend(handles=legend_elements)

    # Make sure values are bounded between 0.5 and 1
    ax.set_ylim(0.5, 1)

    ax.set_xlabel("Subgroup")
    ax.set_ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


def plot_confidence_intervals_from_df(
    df_with_preds,
    label_col,
    pred_col,
    dataset_name,
    group_name,
    use_legend=True,
    figsize=(12, 6),
    fontsize=16,
    tick_spacing=1,
    sort_x=False,
    show_error=True,
    marker="o",
):
    """
    Plots confidence intervals directly from dataframe.
    """
    dict_of_dfs = eval_model_on_groups(
        df=df_with_preds,
        label_col=label_col,
        pred_col=pred_col,
        dataset_name=dataset_name,
        group_names=[group_name],
        print_output=False,
    )

    plot_confidence_intervals(
        dict_of_dfs,
        group_name,
        dataset_name,
        use_legend,
        figsize,
        fontsize,
        tick_spacing,
        sort_x,
        show_error,
        marker,
    )


def plot_intersectional_confidence_intervals_from_df(
    df_with_preds,
    label_col,
    pred_col,
    dataset_name,
    group_1,
    group_2,
    use_legend=True,
    figsize=(12, 6),
    fontsize=16,
    tick_spacing=1,
    sort_x=False,
    show_error=True,
    marker="o",
):
    """
    Plots confidence for overlapping groups.
    """
    intersect_group_name = group_1 + "-" + group_2

    # Copy the df to avoid augmenting the input dataframe in place
    new_df = df_with_preds.copy()
    new_df[intersect_group_name] = (
        df_with_preds[group_1].map(capwords)
        + "-"
        + df_with_preds[group_2].map(capwords)
    )

    # Use existing function on this new group column
    plot_confidence_intervals_from_df(
        new_df,
        label_col,
        pred_col,
        dataset_name,
        intersect_group_name,
        use_legend,
        figsize,
        fontsize,
        tick_spacing,
        sort_x,
        show_error,
        marker,
    )


def plot_confidence_intervals_from_two_dfs(
    df_with_preds_1,
    df_with_preds_2,
    label_col,
    pred_col,
    dataset_name_1,
    dataset_name_2,
    group_name,
    use_legend=False,
    figsize=(12, 6),
    fontsize=16,
    tick_spacing=1,
    show_error=True,
    marker_1="o",
    marker_2="^",
):
    dict_of_dfs_1 = eval_model_on_groups(
        df=df_with_preds_1,
        label_col=label_col,
        pred_col=pred_col,
        dataset_name=dataset_name_1,
        group_names=[group_name],
        print_output=False,
    )

    dict_of_dfs_2 = eval_model_on_groups(
        df=df_with_preds_2,
        label_col=label_col,
        pred_col=pred_col,
        dataset_name=dataset_name_2,
        group_names=[group_name],
        print_output=False,
    )

    plot_dual_confidence_intervals(
        dict_of_dfs_1,
        dict_of_dfs_2,
        group_name,
        dataset_name_1,
        dataset_name_2,
        use_legend,
        figsize,
        fontsize,
        tick_spacing,
        show_error,
        marker_1,
        marker_2,
    )


def compute_wald_intervals(df):
    """
    We are given a dataframe with the columns: 'group', 'correct_count', 'total_count'. For each
    unique group ID, we want one confidence interval that will be centered around correct_count /
    total_count. We will use the Wald approximation since we have a binary samples (correct or
    not).

    Specifically, our formula is p +- z * sqrt(p(1-p) / n) with z = 1.96 for 95% confidence.

    Returns a dict of the form {group_id : (p, epsilon)}
    """
    ci_dict = {}

    for t in df.itertuples():
        group_id, correct_count, total_count = t.group, t.correct_count, t.total_count
        p = correct_count / total_count
        n = total_count
        eps = 1.96 * sqrt(p * (1 - p) / n)
        ci_dict[group_id] = ((p, eps), n)

    return ci_dict
