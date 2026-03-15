import json

# # файл с json
# input_file = "complex_cases.json"
# # выходной txt
# output_file = "complex_cases.txt"

# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# with open(output_file, "w", encoding="utf-8") as f:
#     for item in data:
#         f.write(item["prompt"] + "\n")




# исходный JSON можно читать либо из файла, либо вставить строкой
input_json_path = "simple_cases.json"
output_txt_path = "simple_cases.txt"

with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

prompts = []

# проходим по всем разделам: animals, animals_objects, objects
for items in data.values():
    for item in items:
        prompt = item.get("prompt")
        if prompt:
            prompts.append(prompt)

# записываем только промпты, по одному на строку
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(prompts))

print(f"Готово: сохранено {len(prompts)} промптов в {output_txt_path}")