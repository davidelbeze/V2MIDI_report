import json
from datetime import datetime
import uuid

def parseq_to_deforum(parseq_config):
    def get_max_min_values(keyframes, field):
        values = [kf[field] for kf in keyframes]
        return max(values), min(values)

    with open(parseq_config, 'r') as file:
        data = json.load(file)

    keyframes = data['keyframes']
    rendered_frames_meta = {}
    for field in data['displayFields']:
        max_val, min_val = get_max_min_values(keyframes, field)
        rendered_frames_meta[field] = {"max": max_val, "min": min_val, "isFlat": max_val == min_val}

    rendered_frames = []
    for i, frame in enumerate(keyframes):
        prev_frame = keyframes[i - 1] if i > 0 else {field: 0 for field in data['displayFields']}
        rendered_frame = {
            "frame": frame['frame'],
            "deforum_prompt": f"{data['prompts']['positive']} --neg {data['prompts']['negative']}"
        }
        for field in data['displayFields']:
            delta = frame[field] - prev_frame[field]
            max_val = rendered_frames_meta[field]['max']
            pc = 0 if max_val == 0 else 100 * (frame[field] / max_val)
            rendered_frame[field] = frame[field]
            rendered_frame[f"{field}_delta"] = delta
            rendered_frame[f"{field}_pc"] = pc
        rendered_frames.append(rendered_frame)

    result = {
        "meta": {
            "generated_by": "sd_parseq",
            "version": "0.1.112",
            "generated_at": datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "doc_id": f"doc-{uuid.uuid4()}",
            "version_id": f"version-{uuid.uuid4()}"
        },
        "prompts": {
            "format": "v2",
            "enabled": True,
            "commonPrompt": {
                "name": "Common",
                "positive": "",
                "negative": "",
                "allFrames": True,
                "from": 0,
                "to": keyframes[-1]['frame'],
                "overlap": {
                    "inFrames": 0,
                    "outFrames": 0,
                    "type": "none",
                    "custom": "prompt_weight_1"
                }
            },
            "commonPromptPos": "append",
            "promptList": [
                {
                    "name": "Prompt 1",
                    "positive": data['prompts']['positive'],
                    "negative": data['prompts']['negative'],
                    "allFrames": True,
                    "from": 0,
                    "to": keyframes[-1]['frame'],
                    "overlap": {
                        "inFrames": 0,
                        "outFrames": 0,
                        "type": "none",
                        "custom": "prompt_weight_1"
                    }
                }
            ]
        },
        "options": {**data['options'], "cadence": 1},
        "managedFields": data['displayFields'],
        "displayedFields": data['displayFields'],
        "keyframes": keyframes,
        "timeSeries": [],
        "keyframeLock": "frames",
        "reverseRender": False,
        "rendered_frames": rendered_frames,
        "rendered_frames_meta": rendered_frames_meta
    }

    # Adjust precision for strength.min and strength.max
    if 'strength' in rendered_frames_meta:
        rendered_frames_meta['strength']['max'] = round(rendered_frames_meta['strength']['max'], 14)
        rendered_frames_meta['strength']['min'] = round(rendered_frames_meta['strength']['min'], 14)

    return result

def save_converted_config(input_file, output_file):
    converted_config = parseq_to_deforum(input_file)
    with open(output_file, 'w') as file:
        json.dump(converted_config, file, indent=4)

# Example usage:
# input_file = '/Users/dieny/Desktop/Parseq_Rendered/parseq_config_1.json'
# output_file = '/Users/dieny/Desktop/Parseq_Rendered/parseq_config_1_converted_9.json'
# save_converted_config(input_file, output_file)
