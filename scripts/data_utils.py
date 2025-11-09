import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

def resolve_root_dir():
    env_root = os.environ.get("ROLE_SYNTHCLIP_ROOT")
    if env_root:
        return Path(env_root)

    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "cpfs-data":
            return parent

    for parent in current.parents:
        if (parent / "data").exists() and (parent / "models").exists():
            return parent

    parents = list(current.parents)
    return parents[-1] if parents else current


ROOT_DIR = resolve_root_dir()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

USER_DATA_DIR = DATA_DIR / "huangfuyuanxiang"

def convert_json2csv(paths,output_file_name):
    """
    json:[{
        "image_url": "000000498286.jpg", 只有文件名
        "description": "A close-up of a dog's face, with its eyes closed and nose slightly wrinkled, set against a backdrop of a sunny outdoor scene with a car and a building."
    }]
    csv: title filepath 
    """
    import json
    import os
    data=[]
    if type(paths)==str:
        paths=[paths]
    for path in paths:
        data.extend([{"filepath":  item["image_url"], "title": item["description"]} for item in json.load(open(path))])
    pd.DataFrame(data).to_csv(output_file_name, index=False)
def convert_cocojson2csv(path,output_file_name):
    import json
    import os
    data=json.load(open(path))['annotations']
    ans=[]
    def pad_filename(fn, width=10, padchar='0'):
        fn = str(fn).zfill(width)
        return fn+'.jpg'
    for i,item in enumerate(data):
        ans.append(
            {
                "filepath":pad_filename(item['image_id'],12),
                "title":f"A photo of {item['caption'].strip()}".replace(',','__comma__')
            }
        )
    ans.sort(key=lambda x:x['filepath'])
    from collections import Counter
    counter=Counter([item["filepath"] for item in ans])
    filtered_items = {item: count for item, count in counter.items() if count > 5}
    print(filtered_items)
    pd.DataFrame(ans).to_csv(output_file_name, index=False)
def convert_sharedGPT4V2csv(path,output_file_name):
    import json
    import os
    data=json.load(open(path))
    ans=[]
    for i,item in tqdm(enumerate(data)):
        ans.append(
            {
                "filepath":item['image'],
                "title":f"{item['conversations'][-1]['value']}".replace(',','__comma__')
            }
        )
    pd.DataFrame(ans).to_csv(output_file_name, index=False)    
def convert_flickr2csv(path,output_file_name):
    import json
    import os
    _JSON_KEYS = ['sentids', 'imgid', 'tokens', 'raw', 'sentid']
    ans=[]
    df = pd.read_csv(path)
    for c in _JSON_KEYS:
        df[c] = df[c].apply(json.loads)

    for r_idx, r in df.iterrows():
        r_dict = r.to_dict()
        ans.extend(
            [
                {
                "filepath":r_dict['filename'],
                "title":f"A photo of {r_dict['raw'][i].strip()}".replace(',','__comma__')
            } for i in range(len(r_dict['raw']))]
        )
    pd.DataFrame(ans).to_csv(output_file_name, index=False)
def convert_urban2csv(data_folder,output_file_name):
    data=[]
    file_paths=list(os.listdir(data_folder))
    for path in file_paths:
        img_name=path.split(".txt")[0]+'.jpg'
        title=open(os.path.join(data_folder,path)).readline().strip()
        data.append({
            "filepath":img_name,
            "title":title
        })
    pd.DataFrame(data).to_csv(output_file_name,index=False)
def convert2utf8(json_path):
    import json
    data=json.load(open(json_path,'r',encoding='utf-8'))
    json.dump(data,open(USER_DATA_DIR / "test.json","w",encoding='utf-8'),indent=4, ensure_ascii=False)
if __name__ == "__main__":
    # convert_json2csv([USER_DATA_DIR / "coco2017train_qwenvl_caption_long.json", USER_DATA_DIR / "coco2017train_qwenvl_caption_long_vllm_5.json"],output_file_name="merge_long_train_v2.csv")
    # convert2utf8(USER_DATA_DIR / "TEST_long_vllm.json")
    # convert_cocojson2csv(DATA_DIR / "datasets" / "coco2017" / "annotations" / "captions_val2017.json","coco_official.csv")
    # convert_flickr2csv(DATA_DIR / "datasets" / "Flickr30K" / "test" / "test_1k_flickr.csv","flickr_official.csv")
    # convert_sharedGPT4V2csv(DATA_DIR / "datasets" / "sharedGPT4V" / "share-captioner_coco_lcs_sam_1246k_1107.json",DATA_DIR / "clip_train" / "data" / "sharedGPT4V.csv")
    # convert_urban2csv(DATA_DIR / "datasets" / "Urban1k" / "caption",DATA_DIR / "clip_train" / "data" / "urban_1k.csv")
    
    discussion_root = USER_DATA_DIR / "scripts" / "LLM-Discussion" / "Experiments"
    df1=pd.read_csv(discussion_root / "clean_role_play_p4_VL_long_v3.csv")
    df2=pd.read_csv(discussion_root / "clean_role_play_p4_VL_short_v3.csv")
    df1=pd.concat([df1,df2])
    output_csv = DATA_DIR / "clip_train" / "data" / "shareGPT_VL_roleplay_p4_v3.csv"
    df1.to_csv(output_csv)