import binascii

input_file = "model_ambulance_siren_int8.tflite"
output_file = "model_ambulance_siren_int8.h"
var_name = "model_ambulance_siren_int8"

with open(input_file, "rb") as f:
    data = f.read()

hex_data = ", ".join(f"0x{b:02x}" for b in data)
array_declaration = f"const unsigned char {var_name}[] = {{\n{hex_data}\n}};\n"
array_length = f"const unsigned int {var_name}_len = {len(data)};\n"

with open(output_file, "w") as f:
    f.write(array_declaration)
    f.write(array_length)

print(f"✅ Header file '{output_file}' berhasil dibuat! ({len(data)} bytes)")
