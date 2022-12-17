export TACORL_ROOT=$(pwd)
mkdir -p "$TACORL_ROOT/dependencies/"

# Install CALVIN Env
export CALVIN_ENV_ROOT="$TACORL_ROOT/dependencies/calvin_env"
if [ ! -d "$CALVIN_ENV_ROOT" ] ; then
    git clone --recursive https://github.com/mees/calvin_env.git $CALVIN_ENV_ROOT
    cd "$CALVIN_ENV_ROOT"
else
    cd "$CALVIN_ENV_ROOT"
    git pull --recurse-submodules
fi
cd "$CALVIN_ENV_ROOT/tacto"
pip install -e .
cd "$CALVIN_ENV_ROOT"
pip install -e .
cd ../

# Install R3M
export R3M_ROOT="$TACORL_ROOT/dependencies/r3m"
if [ ! -d "$R3M_ROOT" ] ; then
    git clone https://github.com/facebookresearch/r3m.git $R3M_ROOT
    cd "$R3M_ROOT"
else
    cd "$R3M_ROOT"
    git pull
fi
pip install -e .

# Install TACO-RL
cd $TACORL_ROOT
pip install -e .