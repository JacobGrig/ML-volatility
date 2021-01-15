# ML Volatility
It is a repository with all the code and documentation for the scientific work 
related to the volatility modeling using ML methods. The project is under development, 
and the documentation will be available later.

#### Installation:

- Download and install Anaconda from the following URL:

https://www.anaconda.com/distribution/

Save the path to the installed Anaconda (_path_to_Anaconda_).

- Clone the repository and switch to the tag <tag_name>, using the following command:

`git clone --progress --branch <tag_name> -v "https://github.com/JacobGrig/ML-volatility" _desired_folder_`

- If you use Windows, open Anaconda Prompt, if Linux, use the following command:

`source _path_to_Anaconda_/bin/activate`

- Install _gcc_ (if you have not got it on your computer, only for Linux).

- Create _volatility_ environment by executing the following command:

`conda create --name volatility python=3.9.1`

- Switch to this environment, using the following command:

`conda activate volatility`

- Then go to _desired_folder/ml_volatility_ and execute the following command, 
which installs _ml_volatility_ package and updates _volatility_ environment:

`python python_env_install.py update all`

(instead of the parameter `all` you can also use `packages` to update separately _volatility_ environment, 
or `products` to install separately _ml_volatility_ package).

#### Usage:

- You can run the program, using the following command in Anaconda Prompt:

`ml_volatility run`

and the results will be stored in the folder
_desired_folder/ml_volatility/ml_volatility/data_. 