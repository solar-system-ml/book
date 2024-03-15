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

3. Add changes and commit them.

4. Push the changes to the repository (under the new branch)

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
