from utils.config import types_to_index_group, type_label_to_index_group, type_index_to_label

def fine_tuning (initalcat, type_Id, type, rf_cat, rftype, sub_type, cat, season, style):
    print(initalcat, type_Id, type, rf_cat, rftype, sub_type, cat, season, style)
    
    finaltype = type
    finaltypeid = type_Id
    finalcat = ''
    
    if initalcat != "Shoes" and cat == "Shoes" and rftype == "shoe":
        finaltypeid = types_to_index_group(cat, sub_type)[0]

    if ((type == "KNITWEAR" and cat == "Top" and rftype == "long sleeve")
    or (type != "SKIRTS" and rftype == "skirt")
    or (cat == "Top" and rftype == "hoodie")
    or ("WEAR" in type and cat == "Top" and rftype == "dress")):
        finaltypeid = types_to_index_group("rf", rftype)[0]        

    if type != "ALL-IN-ONES" and initalcat != "Bag" and (rftype == "dress" or rftype == "") and (sub_type == "" or (sub_type != "" and (types_to_index_group(cat, sub_type)[0] == 2 or types_to_index_group(cat, sub_type)[0] != type_Id))):
        finaltypeid = 2        

    if ((initalcat == "Other" or initalcat == "Accessories") or finaltypeid == 0) and types_to_index_group("rf", rftype)[0] > 0:        
        finaltypeid = types_to_index_group("rf", rftype)[0]        
    elif ((initalcat == "Other" or initalcat == "Accessories") or finaltypeid == 0) and cat == rf_cat:
        finaltypeid = types_to_index_group(cat, sub_type)[0]
        
    finaltype = type_index_to_label(finaltypeid)

    if (finaltypeid == 8 # Boots
    or finaltypeid == 33 # Jackets
    or finaltypeid == 74): # Longsleeve
        season = "winter"   
    if finaltypeid == 59: # T-Shirts
        season = "summer"

    if style == "":
        style = 'Smart Casual'
    if season == "":
        season = 'winter/summer'
    
    finalcat = type_label_to_index_group(finaltype)[1]
    print(finalcat, finaltypeid, finaltype, season, style)
    return finalcat, finaltypeid, finaltype, season, style