import math
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def create_cone_model(cone_type, number, x, y, z):
    model = ET.Element("model", name=f"{cone_type}_cone_{number}")
    link = ET.SubElement(model, "link", name="link")
    
    collision = ET.SubElement(link, "collision", name="collision")
    geometry = ET.SubElement(collision, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh, "scale").text = "1 1 1"
    ET.SubElement(mesh, "uri").text = f"../models/cones/{cone_type}cone.dae"
    
    ET.SubElement(collision, "max_contacts").text = "10"
    surface = ET.SubElement(collision, "surface")
    ET.SubElement(surface, "contact").append(ET.Element("ode"))
    ET.SubElement(surface, "bounce")
    friction = ET.SubElement(surface, "friction")
    ET.SubElement(friction, "torsional").append(ET.Element("ode"))
    ET.SubElement(friction, "ode")
    
    visual = ET.SubElement(link, "visual", name="visual")
    geometry = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh, "scale").text = "1 1 1"
    ET.SubElement(mesh, "uri").text = f"../models/cones/{cone_type}cone.dae"
    
    ET.SubElement(link, "self_collide").text = "0"
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "pose").text = "0 0 0 0 -0 0"
    inertia = ET.SubElement(inertial, "inertia")
    ET.SubElement(inertia, "ixx").text = "1"
    ET.SubElement(inertia, "ixy").text = "0"
    ET.SubElement(inertia, "ixz").text = "0"
    ET.SubElement(inertia, "iyy").text = "1"
    ET.SubElement(inertia, "iyz").text = "0"
    ET.SubElement(inertia, "izz").text = "1"
    ET.SubElement(inertial, "mass").text = "1"
    ET.SubElement(link, "enable_wind").text = "0"
    ET.SubElement(link, "kinematic").text = "0"
    
    ET.SubElement(model, "pose").text = f"{x} {y} {z} 0 -0 0"
    
    return model

def generate_world_file(blue_cone_function, yellow_cone_function, num_cones=12, output_file="generated_world.sdf"):
    world = ET.Element("sdf", version="1.7")
    world_elem = ET.SubElement(world, "world", name="default")
    
    # Add blue cones
    for i in range(num_cones):
        angle = 2 * math.pi * i / num_cones
        x, y, z = blue_cone_function(angle)
        world_elem.append(create_cone_model("blue", i, x, y, z))
    
    # Add yellow cones
    for i in range(num_cones):
        angle = 2 * math.pi * i / num_cones
        x, y, z = yellow_cone_function(angle)
        world_elem.append(create_cone_model("yellow", i, x, y, z))
    
    # Create the XML tree and save to file
    xml_str = minidom.parseString(ET.tostring(world)).toprettyxml(indent="  ")
    with open(output_file, "w") as f:
        f.write(xml_str)

# Example usage
def blue_cone_circle(angle, radius=1):
    return radius * math.cos(angle), radius * math.sin(angle), 0.5

def yellow_cone_circle(angle, radius=0.9):
    return radius * math.cos(angle), radius * math.sin(angle), 0.5

generate_world_file(blue_cone_circle, yellow_cone_circle)

print("World file generated successfully!")