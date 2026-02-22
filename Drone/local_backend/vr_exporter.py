"""Export world graph to PLY and generate offline Three.js VR viewer HTML."""

import json
import os
from pathlib import Path
from typing import List, Tuple

from world_graph import WorldGraph


def export_ply(
    points: List[Tuple[float, float, float, float, float, float]],
    path: str,
) -> None:
    """Write point cloud to a PLY file. points: (x,y,z,r,g,b) with r,g,b in 0..1."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt in points:
            x, y, z, r, g, b = pt
            ri = max(0, min(255, int(r * 255)))
            gi = max(0, min(255, int(g * 255)))
            bi = max(0, min(255, int(b * 255)))
            f.write("{} {} {} {} {} {}\n".format(x, y, z, ri, gi, bi))


def export_vr_viewer_html(
    html_path: str,
    ply_filename: str,
    nodes_for_markers: List[dict],
    path_points: List[Tuple[float, float, float]],
) -> None:
    """Generate vr_viewer.html that loads PLY and shows semantic markers. Uses local three.min.js."""
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    # Semantic marker data for Three.js: position [x,y,z], type, label
    markers_js = json.dumps([
        {
            "position": [n.get("x", 0), n.get("y", 0), n.get("z", 0)],
            "type": n.get("type", "clear"),
            "label": n.get("node_id", ""),
        }
        for n in nodes_for_markers
    ])
    path_js = json.dumps(path_points)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>3D Map — VR Viewer</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { overflow: hidden; background: #0a0f19; font-family: sans-serif; }
    #canvas { display: block; width: 100vw; height: 100vh; }
    #hud { position: fixed; left: 12px; top: 12px; color: #eee; font-size: 12px; pointer-events: none; }
    #hud span { display: block; }
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <div id="hud">
    <span>WASD move · Mouse look · R/F up/down</span>
    <span>Point cloud: """ + ply_filename + """</span>
  </div>
  <script src="three.min.js"></script>
  <script>
    (function() {
      var markers = """ + markers_js + """;
      var pathPoints = """ + path_js + """;
      var plyFile = '""" + ply_filename + """';

      var canvas = document.getElementById('canvas');
      var scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0a0f19);
      var camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
      camera.position.set(5, 5, 5);
      var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(window.devicePixelRatio);

      var controls = { move: [0,0,0], yaw: 0, pitch: 0, up: 0, speed: 0.15 };
      document.addEventListener('keydown', function(e) {
        if (e.key === 'w') controls.move[0] = 1;
        if (e.key === 's') controls.move[0] = -1;
        if (e.key === 'a') controls.move[1] = -1;
        if (e.key === 'd') controls.move[1] = 1;
        if (e.key === 'r') controls.up = 1;
        if (e.key === 'f') controls.up = -1;
      });
      document.addEventListener('keyup', function(e) {
        if (e.key === 'w' || e.key === 's') controls.move[0] = 0;
        if (e.key === 'a' || e.key === 'd') controls.move[1] = 0;
        if (e.key === 'r' || e.key === 'f') controls.up = 0;
      });
      document.addEventListener('mousemove', function(e) {
        if (e.buttons) { controls.yaw -= e.movementX * 0.005; controls.pitch -= e.movementY * 0.005; }
      });

      function loadPLY(url, done) {
        var loader = new THREE.FileLoader();
        loader.setResponseType('text');
        loader.load(url, function(text) {
          var lines = text.split('\\n');
          var i = 0;
          while (i < lines.length && !/^element vertex/.test(lines[i])) i++;
          var numVert = parseInt((lines[i].match(/\\d+/) || [0])[0], 10);
          while (i < lines.length && lines[i] !== 'end_header') i++;
          i++;
          var positions = [], colors = [];
          for (var k = 0; k < numVert && i + k < lines.length; k++) {
            var parts = lines[i + k].trim().split(/\\s+/);
            if (parts.length >= 6) {
              positions.push(parseFloat(parts[0]), parseFloat(parts[1]), parseFloat(parts[2]));
              colors.push(parseInt(parts[3],10)/255, parseInt(parts[4],10)/255, parseInt(parts[5],10)/255);
            }
          }
          var geo = new THREE.BufferGeometry();
          geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
          geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
          geo.computeBoundingSphere();
          var mat = new THREE.PointsMaterial({ size: 0.08, vertexColors: true });
          var mesh = new THREE.Points(geo, mat);
          scene.add(mesh);
          if (done) done(mesh);
        }, undefined, function() { if (done) done(null); });
      }

      loadPLY(plyFile, function() {
        // Path line
        if (pathPoints.length >= 2) {
          var pathGeo = new THREE.BufferGeometry().setFromPoints(pathPoints.map(function(p) {
            return new THREE.Vector3(p[0], p[1], p[2]);
          }));
          var pathMat = new THREE.LineBasicMaterial({ color: 0xffdd00, linewidth: 2 });
          scene.add(new THREE.Line(pathGeo, pathMat));
        }
        // Semantic markers as spheres
        var typeColors = { survivor: 0x00ff66, hazard: 0xff3333, exit: 0x3388ff, structural: 0xff8800, clear: 0x888888, unknown: 0x888888 };
        markers.forEach(function(m) {
          var geom = new THREE.SphereGeometry(0.15, 12, 12);
          var col = typeColors[m.type] || 0x888888;
          var mat = new THREE.MeshBasicMaterial({ color: col });
          var sphere = new THREE.Mesh(geom, mat);
          sphere.position.set(m.position[0], m.position[1], m.position[2]);
          scene.add(sphere);
        });
      });

      function animate() {
        requestAnimationFrame(animate);
        var dx = Math.sin(controls.yaw) * controls.move[0] - Math.cos(controls.yaw) * controls.move[1];
        var dz = -Math.cos(controls.yaw) * controls.move[0] - Math.sin(controls.yaw) * controls.move[1];
        camera.position.x += dx * controls.speed;
        camera.position.z += dz * controls.speed;
        camera.position.y += controls.up * controls.speed;
        var look = new THREE.Vector3(
          camera.position.x + Math.sin(controls.yaw) * Math.cos(controls.pitch),
          camera.position.y + Math.sin(controls.pitch),
          camera.position.z - Math.cos(controls.yaw) * Math.cos(controls.pitch)
        );
        camera.lookAt(look);
        renderer.render(scene, camera);
      }
      animate();
      window.addEventListener('resize', function() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      });
    })();
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def run_export(world_graph: WorldGraph, exports_dir: str) -> Tuple[str, str]:
    """Export PLY and VR viewer HTML. Returns (ply_path, html_path). Creates exports_dir if needed."""
    from pointcloud_builder import build_pointcloud

    exports_dir = os.path.abspath(exports_dir)
    os.makedirs(exports_dir, exist_ok=True)
    ply_path = os.path.join(exports_dir, "map.ply")
    html_path = os.path.join(exports_dir, "vr_viewer.html")

    points = build_pointcloud(world_graph)
    export_ply(points, ply_path)

    nodes_for_markers = []
    path_points = []
    if world_graph:
        ordered = sorted(world_graph.nodes.keys())
        for node_id in ordered:
            n = world_graph.nodes[node_id]
            pos = world_graph.get_pose_at_node(node_id)
            if pos is None:
                continue
            x, y, z = pos
            path_points.append((x, y, z))
            # Primary category for marker type
            cats = [d.category.value for d in n.detections]
            node_type = "clear"
            if cats:
                from collections import Counter
                node_type = Counter(cats).most_common(1)[0][0]
            nodes_for_markers.append({
                "node_id": node_id,
                "x": x, "y": y, "z": z,
                "type": node_type,
            })

    export_vr_viewer_html(html_path, "map.ply", nodes_for_markers, path_points)
    return (ply_path, html_path)
