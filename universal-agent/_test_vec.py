import sys, json, inspect
sys.path.insert(0, r'c:\Users\Administrator\Desktop\移动项目\cmos-kb-agent\universal-agent\src')
import kbagent.shared.search as m
print("MODULE FILE:", m.__file__)
from kbagent.shared.search import _vector_raw_to_chunks
print("HAS _info_of:", "_info_of" in inspect.getsource(_vector_raw_to_chunks))

def show(label, chunks):
    print(label, "count=%d" % len(chunks))
    for c in chunks:
        kid = c.position.get("knowledge_id")
        print("  chunk_id=%s doc_id=%s kid=%s title=%s" % (c.chunk_id, c.doc_id, kid, c.doc_title))

# Case 1: knowledgeId nested inside info sub-object
resp1 = {"object": [
    {"info": {"knowledgeId": "K001", "knowledgeName": "5G套餐办理"}, "content": "办理5G套餐的步骤", "score": 0.95},
    {"info": {"knowledgeId": "K002", "knowledgeName": "流量包"}, "content": "流量包详情", "score": 0.8},
]}
show("Case1 nested info:", _vector_raw_to_chunks(resp1, "new"))

# Case 2: knowledgeId at top level (regression)
resp2 = {"object": [
    {"knowledgeId": "K100", "knowledgeName": "top-level", "content": "top content"},
]}
show("Case2 top-level:", _vector_raw_to_chunks(resp2, "old"))

# Case 3: object as JSON string with nested info
resp3 = {"rtnCode": "0", "object": json.dumps([
    {"info": {"knowledgeId": "K200", "knowledgeName": "json-str"}, "content": "json content"},
])}
show("Case3 json-string object:", _vector_raw_to_chunks(resp3, "new"))

# Case 4: top-level info list of entries (existing structure)
resp4 = {"object": {"info": [
    {"knowledgeId": "K300", "knowledgeName": "list-entry", "content": "list content"},
]}}
show("Case4 top-level info list:", _vector_raw_to_chunks(resp4, "old"))
