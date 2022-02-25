import json

def updateJsonFile(compr_model_path, config_path, name_json):
    jsonFile = open(config_path, "r") # Open the JSON file for reading
    data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file

    ## Modify json
    data["pretrainedModel"] = compr_model_path

    ## Save our changes to new JSON file (temp.json)
    jsonFile = open(name_json + ".json", "w+")
    jsonFile.write(json.dumps(data))
    jsonFile.close()
