from dm_control.suite import swimmer

def dump_swimmer_xml(n, out_path):
    xml_bytes, _assets = swimmer.get_model_and_assets(n)   # Calls _make_model(n)
    # xml_bytes is a bytes object; write it verbatim
    with open(out_path, "wb") as f:
        f.write(xml_bytes)

dump_swimmer_xml(6,  "swimmer6.xml")
dump_swimmer_xml(15, "swimmer15.xml")
