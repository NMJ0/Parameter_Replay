
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics,\
    loss_metrics, timing_metrics, cpu_usage_metrics, StreamConfusionMatrix,\
    disk_usage_metrics, gpu_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive
from wandb_logger import WandBLogger

from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import torch 

from fisher_info import compute_empirical_fisher_information,plot_fisher_information_detailed,plot_fisher_information_heatmap


benchmark = SplitMNIST(n_experiences=5,return_task_id=False,seed=42,shuffle=False)   #creates the benchmark
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SimpleMLP(num_classes=benchmark.n_classes).to(device)   #creates the model
print(model)

tb_logger = WandBLogger(project_name="avalanche_tut_!", run_name="run_1") #WandB logger
text_logger = TextLogger(open('log.txt', 'a'))
interactive_logger = InteractiveLogger()


eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    #loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    #timing_metrics(epoch=True),
    #cpu_usage_metrics(experience=True),
    #forgetting_metrics(experience=True, stream=True),
    #StreamConfusionMatrix(num_classes=benchmark.n_classes, save_image=False),
    #disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)

cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=10, eval_mb_size=100,device=device,
    evaluator=eval_plugin)


print('Starting experiment...')
results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    res = cl_strategy.train(experience, num_workers=4)
    print('Training completed')


    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(benchmark.test_stream, num_workers=4))
    print('Computing Fisher Information')
    fisher_dict = compute_empirical_fisher_information(
        model=model,
        dataset=experience.dataset,
        device=device,
        num_samples=200  # Limit to 200 samples for testing
    ) 
    plot_fisher_information_detailed(fisher_dict,model=model,save_path=f'fisher_detailed_exp{experience.current_experience}.png')   
    for name, param in model.named_parameters():
        if name in fisher_dict:
            print(f"Param: {name}, Fisher Info Mean: {fisher_dict[name].mean().item():.6f}")
        else :
            print(f"Param: {name} not found in Fisher Info dictionary.")

torch.save(model.state_dict(), "final_splitmnist_model.pth")
print("Model saved as final_splitmnist_model.pth")
