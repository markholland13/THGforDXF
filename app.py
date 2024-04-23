from flask import Flask, request, jsonify, send_file
import numpy as np
import math
from scipy.special import ellipeinc
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import traceback
import os
import time
from werkzeug.utils import secure_filename

# Assuming bbox.py is in the same directory and contains the Cache and extents functionality
from bbox import Cache, extents

app = Flask(__name__)

# Define geometric calculation functions
def line_length(start, end):
    return np.linalg.norm(np.array(end) - np.array(start))

def circle_length(radius):
    return 2 * math.pi * radius

def arc_length(radius, start_angle_deg, end_angle_deg, is_counter_clockwise=True):
    start_angle_rad = math.radians(start_angle_deg)
    end_angle_rad = math.radians(end_angle_deg)
    if is_counter_clockwise:
        if end_angle_rad < start_angle_rad:
            angular_distance = 2 * math.pi - (start_angle_rad - end_angle_rad)
        else:
            angular_distance = end_angle_rad - start_angle_rad
    else:
        if end_angle_rad > start_angle_rad:
            angular_distance = 2 * math.pi - (end_angle_rad - start_angle_rad)
        else:
            angular_distance = start_angle_rad - end_angle_rad
    return radius * angular_distance

def polyline_length(points):
    total_length = 0
    for i in range(len(points) - 1):
        x1, y1, bulge = points[i]
        x2, y2, _ = points[i + 1]
        dx, dy = x2 - x1, y2 - y1
        chord_length = math.hypot(dx, dy)
        if bulge == 0:
            total_length += chord_length
        else:
            theta = 4 * math.atan(abs(bulge))
            radius = (chord_length / 2) * (math.sqrt(1 + bulge**2) / abs(bulge))
            arc_length_val = radius * theta
            total_length += arc_length_val
    return total_length

def spline_length(spline, num_points=1000):
    t = np.linspace(spline.t[spline.k], spline.t[-spline.k-1], num_points)
    points = spline(t)
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

def ellipse_length(major_axis, ratio, start_angle, end_angle):
    m = 1 - ratio**2
    return major_axis * ellipeinc(end_angle - start_angle, m)

# Handler functions for different DXF entity types
def handle_entity(entity):
    if entity.dxftype() == 'LINE':
        return line_length(entity.dxf.start, entity.dxf.end)
    elif entity.dxftype() == 'CIRCLE':
        return circle_length(entity.dxf.radius)
    elif entity.dxftype() == 'ARC':
        return arc_length(entity.dxf.radius, entity.dxf.start_angle, entity.dxf.end_angle)
    elif entity.dxftype() == 'SPLINE':
        control_points = np.array(entity.control_points)
        knots = np.array(entity.knots)
        degree = entity.dxf.degree
        spline = BSpline(knots, control_points, degree)
        return spline_length(spline)
    elif entity.dxftype() == 'LWPOLYLINE':
        points = entity.get_points(format='xyb')
        return polyline_length(points)
    elif entity.dxftype() == 'ELLIPSE':
        major_axis_endpoint = entity.dxf.major_axis
        major_axis_length = np.linalg.norm(np.array(major_axis_endpoint))
        ratio = entity.dxf.ratio
        start_angle = math.radians(entity.dxf.start_param) if hasattr(entity.dxf, 'start_param') else 0
        end_angle = math.radians(entity.dxf.end_param) if hasattr(entity.dxf, 'end_param') else 2 * math.pi
        return ellipse_length(major_axis_length, ratio, start_angle, end_angle)
    return 0

# DXF to Image conversion class
class DXF2IMG:
    default_img_format = '.png'
    default_img_res = 300

    def convert_dxf2img(self, doc, img_format=default_img_format, img_res=default_img_res):
        msp = doc.modelspace()
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], facecolor='none')
        ax.axis('off')
        ctx = RenderContext(doc)
        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp, finalize=True)
        for patch in ax.patches:
            patch.set_facecolor('white')
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        for line in ax.lines:
            line.set_color('black')
            line.set_linewidth(1)
        img_path = '/tmp/output_image.png'
        fig.savefig(img_path, dpi=img_res, transparent=True)
        plt.close(fig)
        return img_path

# Main processing function
def process_dxf_file(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    total_length = sum(handle_entity(entity) for entity in msp)
    cache = Cache()
    bounding_box = extents(msp, cache=cache)
    converter = DXF2IMG()
    image_path = converter.convert_dxf2img(doc)
    return {
        "total_path_length": total_length,
        "width": bounding_box.size.x,
        "height": bounding_box.size.y,
        "image_path": image_path
    }


@app.route('/upload-dxf', methods=['POST'])
def upload_dxf():
    if 'dxf' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['dxf']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('/tmp', f"{time.time()}_{filename}")  # Use timestamp to avoid filename conflicts
        file.save(filepath)
        try:
            results = process_dxf_file(filepath)
            return jsonify(results)
        except Exception as e:
            os.remove(filepath)  # Cleanup if processing fails
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    return jsonify({'error': 'File processing error'}), 500

@app.route('/get-image/<path:filename>', methods=['GET'])
def get_image(filename):
    image_path = os.path.join('/tmp', secure_filename(filename))  # Secure filename usage
    return send_file(image_path, mimetype='image/png', as_attachment=False, download_name=filename)



if __name__ == '__main__':
    app.run(debug=True)
