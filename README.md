# UniMIND

A Unified Multi-task Learning Framework for Multi-goal Conversational Recommender Systems. 

## Dataset

The TG-ReDial and DuRecDial datasets have been processed to be applied for evaluating multi-goal conversational recommender systems. 

Please cite the following paper if you use it in any way：
    
*	Yang Deng, Wenxuan Zhang, Weiwen Xu, Wenqiang Lei, Tat-Seng Chua, Wai Lam. A Unified Multi-task Learning Framework for Multi-goal Conversational Recommender Systems. 
    	
Also, please cite the original dataset papers following if you use the data:

*	Kun Zhou, Yuanhang Zhou, Wayne Xin Zhao, Xiaoke Wang, Ji-Rong Wen. Towards Topic-Guided Conversational Recommender System. In COLING 2020.
* Zeming Liu, Haifeng Wang, Zheng-Yu Niu, Hua Wu, Wanxiang Che, Ting Liu. Towards Conversational Recommendation over Multi-Type Dialogs. In ACL 2020.


## Training and Inference
`python train.py --do_train --do_finetune --do_pipeline --beam_size=1 --warmup_steps=400 --max_seq_length=512 --max_target_length=100 --gpu=<your_gpu_id> --overwrite_output_dir --per_gpu_train_batch_size=<your_batch_size> --per_gpu_eval_batch_size=<your_batch_size> --model_name_or_path="fnlp/bart-base-chinese" --data_name=<tgredial or durecdial>`
