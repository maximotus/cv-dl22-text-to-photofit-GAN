# TODOs

## Submission folder should contain
- [ ] final code as zip
- [ ] report as pdf
- [ ] slides as pdf
- [ ] presentation as mp4

## Prio 0

- [ ] setup of cuda_home in README.md or out-comment the parts in helper.py and tedi_gan.py that are not working for the final submission
- [ ] include example vector and its output to report and presentation (because it is our main part of the project)
- [ ] check executability of the whole framework
  - [x] train
  - [ ] eval (see (a)) @SchubertDaniel
  - [x] gen
- [ ] double check / read report MAX
- [ ] double check / read report DANIEL
- [ ] mention in the experiments section that we only useed CDCGAN and not tediGAN (so its more clear, even if it could be clear from context in tedigan section)
- [ ] make video of presentation
- [x] language english in report / latex (e.g. "Abbildung 8")
- [x] print CDCGAN architectures and insert to report
- [x] implement plot script for plotting stats for report
- [x] update README.md with all necessary information to run our framework
- [x] write report
- [x] create presentation slides
- [x] rename LSW to LFW

## Prio 1

- [ ] eliminate duplicated code lines (e.g. super class for a PhotofitGeneratorModel or GAN or so)
- [ ] use string format instead of concatenations with +
- [ ] future work: extend framework and eliminate duplicated code in tediGAN and CDCGAN (e.g. b using a parent class GAN)
- [ ] implement save_plots (save plots as pngs) at the end of a train run
- [x] do continuation in main, do not use extra class for that (e.g. after line 79)
- [x] rename current_epoch to start_epoch
- [x] pass pretrained_path and start_epoch as model parameters
- [x] check in model if these parameters are None, if not, initialize pretrained model --> otherwise one would need an extra continuation class for each model of the framework (for future extensions)
- [x] rename dataset_size_factor to dataset_size_fraction
- [x] refactoring of the program structure (packages, modules)
- [x] extract all valid names to one file
- [x] refactor the config file scheme
- [x] rename TRAINED_PATH to PRETRAINED_PATH_KEY
- [x] rename CURRENT_EPOCH to CURRENT_EPOCH_KEY
- [x] epochs is not max_epochs, but rather the number of remaining epochs. so we can just do range(epochs) instead of range(current_epoch + 1, epochs)
- [x] adapt CDCGAN.generate_image method to be able to generate multiple images

(a)
```
Traceback (most recent call last):
  File "C:\Users\Max\miniconda3\envs\deep_learning\lib\runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\Users\Max\miniconda3\envs\deep_learning\lib\runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "C:\Users\Max\miniconda3\envs\deep_learning\lib\site-packages\pytorch_fid\__main__.py", line 3, in <module>
    pytorch_fid.fid_score.main()
  File "C:\Users\Max\miniconda3\envs\deep_learning\lib\site-packages\pytorch_fid\fid_score.py", line 274, in main
    num_avail_cpus = len(os.sched_getaffinity(0))
AttributeError: module 'os' has no attribute 'sched_getaffinity'
Traceback (most recent call last):
  File "C:\Users\Max\Documents\bildung\uni\master_lmu\semester_1\computer_vision_and_deep_learning\cv-dl22-text-to-photofit-GAN\src\main.py", line 85, in <module>
    evaluator.evaluate()
  File "C:\Users\Max\Documents\bildung\uni\master_lmu\semester_1\computer_vision_and_deep_learning\cv-dl22-text-to-photofit-GAN\src\training.py", line 155, in evaluate
    logger.info(scores)
  File "C:\Users\Max\Documents\bildung\uni\master_lmu\semester_1\computer_vision_and_deep_learning\cv-dl22-text-to-photofit-GAN\src\metric.py", line 14, in FID
    x=subprocess.check_output(queue, shell=True)
  File "C:\Users\Max\miniconda3\envs\deep_learning\lib\subprocess.py", line 424, in check_output
    return run(*popenargs, stdout=PIPE, timeout=timeout, check=True,
  File "C:\Users\Max\miniconda3\envs\deep_learning\lib\subprocess.py", line 528, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command 'python -m pytorch_fid ../experiments/eval/template-CDCGAN-train/2022-08-12-09-56-05//img-fake ../experiments/eval/template-CDCGAN-train/2022-08-12-09-56-05//img-real' returned non-zero exit status 1.
```