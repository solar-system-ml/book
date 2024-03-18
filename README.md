# Supplementary Materials

## How to make changes

1. Clone the repository

```bash
git clone https://github.com/solar-system-ml/book.git
```

2. Set up a new branch

```bash
git checkout -b mybranch
```

NB: use your branch name instead of `mybranch`.

3. Add changes and commit them.

4. Push the changes to the repository (under the new branch)

```bash
git push -u origin mybranch
```

5. If you know how to do this, create pull request to the `main` branch. Otherwise, simply send an email to Evgeny to incorporate the changes.

## How to run the project locally?

First of all, there is no particular need to install the project locally. You can only add your notebooks to the proper folder and commit the changes. However, if you want to check how will it be look like, follow the steps below.

Firstly, you need to install all requirements.

```bash
pip install -r requirements.txt
```

Secodly, you need to run the `mkdocs`:

```bash
mkdocs serve
```

It will run the documentation locally and make it available by the following URL: `http://127.0.0.1:8000/`

If you need to add a navigation item (i.e., link to your notebook or chapter), you need to edit `mkdocs.yml` file, the section called `nav`. It utilises the key-value list. The keys are the titles of the links, the values are the files. The root directory is `docs`. Please refer to the example of `chapter3/example.ipynb` available there.

## Saving outputs

Note that when you save a noteboook, usually, it saves all the output produced including data, images, and other information. If you produce multiple images, it might require a lot of storage, which is not the case for GitHub. If the amount of images is less than 10-20, then it is fine. However, if your notebook produces 50-100+ images, it would be better to use one of the approaches below:

1. Save some outputs and do not save the rest. A reader can run these notebooks by herself and get all images locally.
1. Create several notebooks and split images between them. Some notebooks can have outputs, whereas some can contain only inputs.
1. Do not produce images in notebooks and upload them to the external file storage and add a link to the notebook.

Any approach from above is good. You are free to use any other solution that does not require to store a lot of data.

## Commands

-   `mkdocs serve` - Start the live-reloading docs server.
-   `mkdocs build` - Build the documentation site.
-   `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        chapter*  # Chapter folders and pages. Place there Jupyter notebooks and markdown files.
        ...       # Other markdown pages, images and other files.
