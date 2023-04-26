

# Demonstration 

- pydantic pyfig
    - nested
    







# TODO
- sweep / hypam_opt dict

# baselines

- [https://github.com/Yunbo426/MIM](https://github.com/Yunbo426/MIM)
- [https://github.com/alexlee-gk/video_prediction](https://github.com/alexlee-gk/video_prediction)
- [https://github.com/edenton/svg](https://github.com/edenton/svg)
https://attractors.pyviz.demo.anaconda.com/attractors_panel
- https://unit8co.github.io/darts/userguide.html
- https://jupyterbook.org/en/stable/interactive/interactive.html

# this is some expxloding brain level shit
https://stackoverflow.com/questions/72080471/cant-install-darts-llvmlite-on-mac-python-3-8-9/73701726#73701726

pip install CMAKE
pip install setuptools
pip3 install gensim
brew install lightgbm
brew install cmake
brew install libomp
pip install httpstan
pip install lightgbm
pip install ipython
pip install google-auth-oauthlib
pip install filterpy
pip install cmdstanpy
pip install tensorboard
pip install prophet
pip install pmdarima
pip install darts
pip list | grep darts

# RunFlow

- Add _user.py file with your details
    - user = your server user id

https://github.com/facebook/prophet/issues/2002

python 3.8.13
cmdstanpy 0.9.68
prophet 1.0.1
pystan 2.19.1.1
Prophet 1.1 does not seem compatible with python 3.10 with the M1 chip.

# Doc
## How to Run
###### #pyfig-recurse
Calling properties recursively is bad, exclude some properties from config dictionary to avoid this. 

###### #_n_submit-state-flow
initial -1 ( into pyfig.submit() )
changed to number of submissions ( into pyfig.__init__, is positive 1 or n_sweep, exits after submission )
changed to zero (on cluster submit)


- hypam opt
    - opt: adahess, radam
    - cudnn: T, F
    - lr: sample
    - hessian_power: sample
    - n_b: 1024

- hypam opt
    - n_sv
    - n_pv
    - n_fb
    - n_det


https://wandb.ai/hwat/hwat/groups/dist/workspace
https://wandb.ai/hwat/hwat/groups/sweep-a_z/workspace
https://wandb.ai/hwat/hwat/groups/sweep-n_b/workspace
https://wandb.ai/hwat/hwat/groups/hypam_opt/workspace

baselines :exclamation:
memory scaling and gpu usage scaling (1 gpu)
thread testing pytorch :exclamation:
optimise hyperparams
scale across 1 gpu n_e max_mem opt
scaling across gpus (naive)
scaling across gpus (accelerate)
(including nodes)
take those wandb reports and copy the application and send to eske
message Steen 8 node 4h block







# Docs
## Rules
### 


## How to run
### Definitions
- Local (your laptop or a cloud development environment)
- Server (the stepping stone to cluster)
- Cluster (where the magic happens)

### Commands
- jupyter nbconvert --execute <notebook>

### Local -> Server -> Cluster
- 

### Server -> Cluster
- 

# FIX 
- pre on diag gaussian
- Loop over dense
- Do arrays roll from right


# How To
- Fill in User Details in Pyfig


# Doc Notes
## Glossary


# Test Suite
- Anti-symmetry

# Theory notes
- Why is the mean over the electron axes equal to a quarter? 
    - The mean of the entire thing is equal to zero...
    - * this problem is interesting and lead to the tril idea

# Model Ideas
- Alter the dims of the embedded variables to match the first FermiBlock layer, giving more residual connections (ie mean with keepdims)
    - name: 'fb0_res'
    - r_s_res = r_s_var.mean(-1, keepdims=True) if _i.fb0_res else jnp.zeros((_i.n_e, _i.n_sv), dtype=r.dtype)
- Electrons are indistinguishable... Why no mix mas? Eg in the initial layers, extrac the mean out the fermi block and perform it every iteration removing the means from the fermi block 
- Tril to drop the lower triangle duh? 
    - Need to check the antisymmetry, for sure
- Max pools ARE ALSO PERMUTATION INVARIANT
- Keep the atom dimension so perform ops?
- To test 'only upper triangle' - tril the inputs
- Test limiting the log_psi esp early in training (regularise)


# Gist / Notion / Embed / Share
- https://blog.shorouk.dev/2020/06/how-to-embed-any-number-of-html-widgets-snippets-into-notion-app-for-free/

https://hwat.herokuapp.com/panel_demo


# Setup
## Requirements file
- pipreqsnb <jup-notebook>
- pipreqs <python-file>

## Procfile
### Abstract
- <indicate-what-kind-of-app-as-defined-by-heroku>: <a-cmd-to-run-the-app>
### Generalisation
- web: panel serve --address="0.0.0.0" --port=$PORT iris_kmeans.ipynb --allow-websocket-origin=hwat.herokuapp.com
### Description
- web: panel serve --address="0.0.0.0" --port=$PORT iris_kmeans.ipynb --allow-websocket-origin=hwat.herokuapp.com
### Example
- web: panel serve --address="0.0.0.0" --port=$PORT iris_kmeans.ipynb --allow-websocket-origin=hwat.herokuapp.com

# Heroku
## tar install (NO MARCHA)
"""
wget https://cli-assets.heroku.com/branches/stable/heroku-OS-ARCH.tar.gz
tar -xvzf heroku-OS-ARCH -C /usr/local/lib/heroku
ln -s /usr/local/lib/heroku/bin/heroku /usr/local/bin/heroku
"""

## Launch app
- heroku login
- heroku create <app-name>
- git push heroku master
    - git push heroku main
- heroku create -a example-app [auto adds heroku remote]
- git remote -v (checks ]
- app_exists: 
    - heroku git:remote -a example-app
- git push heroku main

## ide-yeet 
- Other 


<script src="https://gist.github.com/xmax1/f9f66535467ec44759193a18594e72c4.js"></script>