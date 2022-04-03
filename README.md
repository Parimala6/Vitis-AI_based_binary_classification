# Eye state detection using Vitis-AI
## Vitis-AI version - 1.4.0
### Tested on KV260

Binary classification model summary: <br/>
<img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/model.PNG" width="400" height="400">

Training and validation accuracy:
<img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/train_acc.PNG" width="800" height="300">

#### Model performance on KV260 
<table>
    <thead>
        <tr>
            <th> DPU version </th>
            <th> Threads = 1 </th>
            <th> Threads = 2 </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> B3136 </td>
            <td> <img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/b3136_t1.JPG"> </td>
            <td> <img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/b3136_t2.JPG"> </td>
        </tr>
        <tr>
            <td> B4096 </td>
            <td> <img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/b4096_t1.JPG"> </td>
            <td> <img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/b4096_t2.JPG"> </td>
        </tr>
    </tbody>
</table>

#### KV260 performance while running the model
<table>
    <thead>
        <tr>
            <th> DPU version </th>
            <th> Threads = 1 </th>
            <th> Threads = 2 </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> B3136 </td>
            <td> <img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/b3136_stats_t1.JPG"> </td>
            <td> <img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/b3136_stats_t2.JPG"> </td>
        </tr>
        <tr>
            <td> B4096 </td>
            <td> <img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/b4096_stats_t1.JPG"> </td>
            <td> <img src="https://github.com/Parimala6/Vitis-AI_based_binary_classification/blob/main/images/b4096_stats_t2.JPG"> </td>
        </tr>
    </tbody>
</table>

For the detailed guide on how to run the code refer to - https://www.hackster.io/Parimala6/eye-state-detection-model-implementation-on-kria-3415a3
