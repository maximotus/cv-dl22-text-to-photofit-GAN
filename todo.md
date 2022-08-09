# TODOs

## Prio 0

- [ ] implement plot script for plotting stats for report
- [ ] implement save_plots (save plots as pngs) at the end of a train run
- [ ] update README.md with all necessary information to run our framework
- [ ] write report
- [ ] create presentation slides
- [ ] make video

## Prio 1

- [x] do continuation in main, do not use extra class for that (e.g. after line 79)
- [x] rename current_epoch to start_epoch
- [x] pass pretrained_path and start_epoch as model parameters
- [x] check in model if these parameters are None, if not, initialize pretrained model --> otherwise one would need an extra continuation class for each model of the framework (for future extensions)
- [x] rename dataset_size_factor to dataset_size_fraction
- [x] refactoring of the program structure (packages, modules)
- [x] extract all valid names to one file
- [x] refactor the config file scheme
- [ ] eliminate duplicated code lines (e.g. super class for a PhotofitGeneratorModel or GAN or so)
- [ ] use string format instead of concatenations with +
- [x] rename TRAINED_PATH to PRETRAINED_PATH_KEY
- [x] rename CURRENT_EPOCH to CURRENT_EPOCH_KEY
- [x] epochs is not max_epochs, but rather the number of remaining epochs. so we can just do range(epochs) instead of range(current_epoch + 1, epochs)