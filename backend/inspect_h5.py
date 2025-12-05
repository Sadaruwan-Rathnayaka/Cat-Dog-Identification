# inspect_h5.py
import h5py, json, sys
p = sys.argv[1] if len(sys.argv)>1 else "saved_model.h5"
print("Checking:", p)
f = h5py.File(p, "r")
print("Top-level groups:", list(f.keys()))
print("Attrs keys:", list(f.attrs.keys()))
if 'model_config' in f.attrs:
    cfg = f.attrs['model_config']
    cfg_str = cfg.decode('utf-8') if isinstance(cfg,(bytes,bytearray)) else cfg
    j = json.loads(cfg_str)
    layers = j.get('config', {}).get('layers', [])
    print("Layer count in model_config:", len(layers))
    for L in layers:
        print(L.get('class_name'), "-", L.get('config',{}).get('name'))
else:
    # show groups under model_weights if present
    if 'model_weights' in f:
        print("model_weights groups:", list(f['model_weights'].keys())[:50])
    else:
        print("No model_config attr and no model_weights group found.")
f.close()
