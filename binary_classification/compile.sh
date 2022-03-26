compile() {
    vai_c_tensorflow \
	--frozen_pb ./quantize_results/quantize_eval_model.pb \
	--arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json \
	--output_dir ./output/ \
	--net_name binary_classification \
	--options "{'mode':'normal'}" 
}

compile | tee ./log/compile_log_kv260
