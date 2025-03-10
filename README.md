# test-time-FT

How to solve optimization problems with natural language constraints via test time training

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```
LICENSE
README.md
generate_readme.sh
requirements.txt
setup_env.sh
src/__init__.py
src/algs.py
src/configs/config.json
src/data/__init__.py
src/data/data_utils.py
src/generate_plots.py
src/models/__init__.py
src/models/model.py
src/run_experiment.py
src/train/train.py
```

## Installation

This project requires Python and the dependencies listed in `requirements.txt`.

### Automatic Setup

Run the setup script to create a virtual environment and install dependencies:

```bash
./setup_env.sh
```

### Manual Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - On Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. After setting up the environment, you can edit the config.json file depending on which problem and algorithm you want to run. [View Configuration Documentation](https://pranjal-awasthi.github.io/test-time-FT/config_doc.html) 

2. Run:
``` python run_experiment.py --config location_of_config_file --seed random_seed
```

The above command will generate a random experiment id and store all the evals in a corresponding directory. To generate the plots run:

```
python generate_plot.py --experiment_dir experiment_dir --experiment_id experiment_id
```

<!-- Add usage instructions here -->


## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Pranjal Awasthi - [GitHub](https://github.com/pranjal-awasthi)

