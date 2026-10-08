---
jupyter:
  jupytext:
    default_lexer: python
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Homework 1

```python
print("Hello from michaels home computer!!\nBelow are the answers to the homework")
```

# Question 1

The key to managing files between github and google colab is locally maintaining the files. You 
mention using vscode but as explained later, my text editor of choice is neovim. This is a bit
harder to configure but it's a hobby of mine so it works great for me and I get to 
keep the environment I am used to.

# Question 2

The only problems I had with executing the code from the modules was the python modules needed
as explained in the next question. My configuration for my setup is terminal based so i needed all
of the python modules from pip to run the code. This just included things like seaborn, pandas, 
matplotlib, and scikit-learn. In only took a minute or two to get everything running but now it works fine

# Question 3

The setup I currently have running revolves around neovim in my terminal. I have always preferred configuring
my terminal to run things locally. It allows me to have a better understanding of what is going in with the
system. My current system revoles around having certain plugins in my terminal. One of the
main platers is the python virtuen environment. This allows me to install python modules on my 
mac outside the scope of my brew install so I don't mess up the rest of the packages installed
on my machine. It's also easily managed as it only runs in the dir I direct it to and it turns
off automatically when I close my terminal. Some other packages are molten.nvim and 
jupytext. These just allow me to convert the raw json of the notebooks into a .py file so it's readable.
This all revoles around a local server running on my machine to host jupytext which acts kind of like a 
parser to translate the jupyter notebook json config file into an editable notebook. Otherwise, I use the terminal
git command to clone a repo and then just make my own and upload the changes to my github account. I am able
to do this through an ssh key generated from openssh. Basically just allowing me to have a tunnel into my github
account through my machine. I use google colab to basically make sure everything is running fine. But the jupyter
notebook files generate just fine locally.

# Question 4

Understanding the basics of python is important because the basic operations on certain data structures and the understanding of such things
is much needed for something like machine learning. If you don't understand what your language can do, how will you ever
get work done? Also, understanding the libraries you will be using is important so you aren't wondering what is
going on when you are implementing certain things. It will smooth things over later. The hardest part is remembering the individual
methods in my opinion. It's like learning anything in the CS world. You can understand the theory but practicality comes with time.
