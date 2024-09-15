# 8.	Resources

## 8.1.	Getting Python packages on MATLAB Online
MATLAB Online is a great pre-configured environment to demo MATLAB with Python. And I’m not just saying this now that I’ve joined the MATLAB Online product team (mid 2023). I have been using it since 2022 to run workshops with Heather, either for large public events like MATLAB EXPO, or for customer dedicated hands-on workshops. This avoids wasting the first half hour of any conversation in setting up the MATLAB & Python environments (see chapter 3), as Python 3 is pre-installed there.
But there’s a catch: what always prevented me from using this nice online environment productively for bilingual workflows was the lack of Python packages, and a way to customize the environment. I obviously tried to find ways around this limitation, by uploading the packages sources as zipped files, and unzip it in MATLAB Online. However, for foundational packages like Numpy, it was already amounting for over 7,000 files to write to disk (only 50 Mb uncompressed). So, it takes a while…
Then recently, after discussing with my colleagues in the MATLAB Online team, they suggested a very simple approach to retrieve pip from a single script:
```matlab
>> websave("get-pip.py","https://bootstrap.pypa.io/get-pip.py");
>> !python get-pip.py
>> !python -m pip --version
pip 24.2 from /home/matlab/.local/lib/python3.10/site-packages/pip (python 3.10)
```
You can now simply install a package like numpy as such:
```matlab
>> !python -m pip install numpy
```
Bear in mind that the MATLAB Online environment is ephemeral, and that you will have to repeat this process each time you start a new session.

## 8.2. Generate this book with Quarto and Pandoc

This book is written with different editing tools: Word, Live scripts, , Jupyter Notebook and Markdown. In order to maintain consistency and automate the generation of the different formats, different conversions are applied:
- live scripts > word 
- live scripts > markdown
- live scripts > jupyter notebook
- jupyter notebook > markdown
- jupyter notebook > word

I am leveraging an open-source software called [Quarto](https://quarto.org/), developed by Posit (formerly known as the company RStudio). Quarto is built on [Pandoc](https://pandoc.org/) a universal document converter that allows you to write in Markdown, and generate a book in multiple formats, including Word, PDF, HTML, Markdown, and Jupyter Notebooks. 

Quarto is a great tool to write technical books, as it complements Pandoc by supporting code in multiple languages, including MATLAB and Python, with interactive code snippets, that can be executed in VS Code or Jupyter.

Here is one example of a command to generate word documents from the Jupyter notebooks:

```bash
quarto pandoc notebook.ipynb -s -o README.md
```