cd universal-agent

python scripts/run_processing_demo.py `
    --input-json "demo-inputs/real_processing_input.json" `
    --query "流量查询" `
    --retrieval-query "流量查询" `
    --region-id "200" `
    --region-name "广东" `
    --channel-code "1" `
    --output-dir "demo-output/real-processing3"