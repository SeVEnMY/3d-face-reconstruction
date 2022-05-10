import importlib
import torch
import os
import time
from util.visualizer import MyVisualizer


def main(rank, word_size):

    epoch_count = 1
    n_epochs = 20
    print_freq = 100
    display_freq = 1000
    evaluation_freq = 5000
    add_image = True
    vis_batch_nums = 1
    batch_size = 32
    eval_batch_nums = float('inf')
    save_latest_freq = 5000
    name = 'face_recon'
    save_by_iter = True
    save_epoch_freq = 1

    device = torch.device(rank)
    torch.cuda.set_device(device)
    

    train_dataset, val_dataset = create_dataset(train_opt, rank=rank), create_dataset(val_opt, rank=rank)
    train_dataset_batches, val_dataset_batches = \
        len(train_dataset) // 32, len(val_dataset) // 32
    
    # Import data set
    dataset_name = "./datasets/example"
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls
    
    dataset_class = dataset
    sampler = 
    dataloader = torch.utils.data.DataLoader(dataset_class, sampler=None, num_workers=4/word_size, batch_size=32/word_size,drop_last=True)

    model_name = 'facerecon'
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls
    
 
    total_iters = train_dataset_batches * (20 - 1)   # the total number of training iterations
    t_data = 0
    t_val = 0
    optimize_time = 0.1
    batch_size = 32

    times = []
    for epoch in range(epoch_count, n_epochs + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for train_data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        train_dataset.set_epoch(epoch)
        for i, train_data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += batch_size
            epoch_iter += batch_size

            torch.cuda.synchronize()
            optimize_start_time = time.time()

            model.set_input(train_data)  # unpack train_data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if rank == 0 and (total_iters == batch_size or total_iters % display_freq == 0):   # display images on visdom and save images to a HTML file
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), total_iters, epoch,
                    save_results=True,
                    add_image=add_image)
            
            if rank == 0 and (total_iters == batch_size or total_iters % print_freq == 0):    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters == batch_size or total_iters % evaluation_freq == 0:
                with torch.no_grad():
                    torch.cuda.synchronize()
                    val_start_time = time.time()
                    losses_avg = {}
                    model.eval()
                    for j, val_data in enumerate(val_dataset):
                        model.set_input(val_data)
                        model.optimize_parameters(isTrain=False)
                        if rank == 0 and j < vis_batch_nums:
                            model.compute_visuals()
                            visualizer.display_current_results(model.get_current_visuals(), total_iters, epoch,
                                    dataset='val', save_results=True, count=j * batch_size,
                                    add_image=add_image)

                        if j < eval_batch_nums:
                            losses = model.get_current_losses()
                            for key, value in losses.items():
                                losses_avg[key] = losses_avg.get(key, 0) + value

                    for key, value in losses_avg.items():
                        losses_avg[key] = value / min(eval_batch_nums, val_dataset_batches)

                    torch.cuda.synchronize()
                    eval_time = time.time() - val_start_time
                    
                    if rank == 0:
                        visualizer.print_current_losses(epoch, epoch_iter, losses_avg, eval_time, t_data, dataset='val') # visualize training results
                        visualizer.plot_current_losses(total_iters, losses_avg, dataset='val')
                model.train()      

            if rank == 0 and (total_iters == batch_size or total_iters % save_latest_freq == 0):   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if save_by_iter else 'latest'
                model.save_networks(save_suffix)
            
            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, n_epochs, time.time() - epoch_start_time))
        
        if rank == 0 and epoch % save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore")
           
    main(0, 1)