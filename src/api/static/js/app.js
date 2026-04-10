/**
 * ExploreSat – Core Application Logic
 */

const API = ''; // same-origin

class ExploreSatApp {
  constructor() {
    this.map = null;
    this.drawnItems = new L.FeatureGroup();
    this.currentBBox = null;
    this.activeTileLayers = {};
    this.layerControl = null;

    this.init();
    this.setupEventListeners();
  }

  init() {
    // Initialize Map
    this.map = L.map('map', {
      zoomControl: false,
      attributionControl: false
    }).setView([20.5937, 78.9629], 5);

    L.control.zoom({ position: 'bottomright' }).addTo(this.map);
    L.control.attribution({ position: 'bottomright' }).addTo(this.map);

    // Base layers
    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
    }).addTo(this.map);

    const esriSat = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18 }
    );

    const baseLayers = { 'OpenStreetMap': osm, 'Satellite (Esri)': esriSat };
    this.layerControl = L.control.layers(baseLayers, {}, { position: 'bottomright' }).addTo(this.map);

    // Draw control
    this.drawnItems.addTo(this.map);
    const drawControl = new L.Control.Draw({
      position: 'topright',
      draw: {
        rectangle: { shapeOptions: { color: '#00f2ff', weight: 2 } },
        polygon: false, circle: false, marker: false, polyline: false, circlemarker: false
      },
      edit: { featureGroup: this.drawnItems },
    });
    this.map.addControl(drawControl);

    this.map.on(L.Draw.Event.CREATED, (e) => {
      this.drawnItems.clearLayers();
      this.drawnItems.addLayer(e.layer);
      const b = e.layer.getBounds();
      this.currentBBox = {
        lon_min: b.getWest(), lat_min: b.getSouth(),
        lon_max: b.getEast(), lat_max: b.getNorth(),
      };
      this.setStatus('dl-status', `AOI set successfully`, 'ok');
    });

    // Legend
    this.addLegend();

    // Initial layer refresh
    this.refreshLayers();

    // Initial layer refresh
    this.refreshLayers();
  }

  addLegend() {
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = () => {
      const div = L.DomUtil.create('div', 'glass-card legend');
      const classes = [
        ['#000000', 'No Data'],
        ['#0000ff', 'Water'],
        ['#d2b48c', 'Natural Bare Ground'],
        ['#ff0000', 'Artificial Bare Ground'],
        ['#006400', 'Woody Vegetation'],
        ['#228b22', 'Cultivated Vegetation'],
        ['#9acd32', '(Semi) Natural Vegetation'],
        ['#ffffff', 'Permanent Snow/Ice'],
      ];
      div.innerHTML = '<strong style="color:var(--accent-primary);display:block;margin-bottom:8px;font-size:0.85rem;">Classes</strong>' +
        classes.map(([c, n]) =>
          `<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:0.75rem;">
            <span style="width:12px;height:12px;border-radius:2px;background:${c};border:1px solid rgba(255,255,255,0.2);"></span>
            ${n}
          </div>`
        ).join('');
      return div;
    };
    legend.addTo(this.map);
  }

  setStatus(id, msg, type) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = msg;
    el.className = 'status-msg active ' + (type || '');
    if (type === 'ok') {
      setTimeout(() => el.classList.remove('active'), 5000);
    }
  }

  setupEventListeners() {
    // Global exposure for HTML onclicks (keeping it simple for refactor)
    window.downloadData = () => this.downloadData();
    window.runInference = () => this.runInference();
    window.refreshLayers = () => this.refreshLayers();
    window.toggleLayer = (name, checked) => this.toggleLayer(name, checked);
  }

  async downloadData() {
    const bbox = this.currentBBox || { lon_min: 77.5, lat_min: 28.5, lon_max: 77.8, lat_max: 28.8 };
    const body = {
      backend: document.getElementById('dl-backend').value,
      dataset: document.getElementById('dl-dataset').value,
      bbox,
      date_start: document.getElementById('dl-start').value,
      date_end: document.getElementById('dl-end').value,
      max_cloud_cover: parseFloat(document.getElementById('dl-cloud').value),
      gee_project: document.getElementById('dl-gee-project').value || null,
    };

    this.setStatus('dl-status', 'Processing download request...', '');
    try {
      const res = await fetch(`${API}/download/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      this.setStatus('dl-status', data.message, res.ok ? 'ok' : 'err');
    } catch (e) {
      this.setStatus('dl-status', 'Error: ' + e.message, 'err');
    }
  }

  async runInference() {
    const fileInput = document.getElementById('infer-file');
    if (!fileInput.files.length) {
      this.setStatus('infer-status', 'Select an image file.', 'err');
      return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    const exportVec = document.getElementById('infer-vector').checked;

    this.setStatus('infer-status', 'Running AI feature extraction...', '');
    try {
      const res = await fetch(`${API}/inference/?export_vector=${exportVec}`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        this.setStatus('infer-status', data.detail || 'Extraction failed', 'err');
        return;
      }
      this.setStatus('infer-status', `Extraction complete.`, 'ok');
      const layerName = `${data.job_id}_pred_rgb`;
      this.addTileLayer(layerName);
      this.refreshLayers();
    } catch (e) {
      this.setStatus('infer-status', 'Error: ' + e.message, 'err');
    }
  }

  async refreshLayers() {
    try {
      const res = await fetch(`${API}/tiles/layers`);
      const data = await res.json();
      this.renderLayerList(data.layers || []);
    } catch (e) {
      document.getElementById('layer-list').innerHTML =
        `<p style="color:var(--accent-secondary);font-size:0.75rem;">Source unavailable.</p>`;
    }
  }

  renderLayerList(layers) {
    const container = document.getElementById('layer-list');
    if (!layers.length) {
      container.innerHTML = '<p style="font-size:0.75rem;color:var(--text-secondary);">No layers found.</p>';
      return;
    }
    container.innerHTML = layers.map(l => `
      <div class="layer-item">
        <div class="layer-info">
          <input type="checkbox" id="chk-${l}" onchange="toggleLayer('${l}', this.checked)" ${this.activeTileLayers[l] ? 'checked' : ''} style="width:auto;">
          <label for="chk-${l}" style="margin:0;cursor:pointer;" class="layer-name">${l}</label>
        </div>
      </div>
    `).join('');
  }

  toggleLayer(name, checked) {
    if (checked) {
      this.addTileLayer(name);
    } else {
      if (this.activeTileLayers[name]) {
        this.map.removeLayer(this.activeTileLayers[name]);
        this.layerControl.removeLayer(this.activeTileLayers[name]);
        delete this.activeTileLayers[name];
      }
    }
  }

  addTileLayer(name) {
    if (this.activeTileLayers[name]) return;
    const tl = L.tileLayer(`${API}/tiles/${name}/{z}/{x}/{y}.png`, {
      attribution: 'ExploreSat Prediction',
      opacity: 0.8,
      maxZoom: 21,
      tileSize: 256,
    }).addTo(this.map);
    this.activeTileLayers[name] = tl;
    this.layerControl.addOverlay(tl, name);
  }
}

// Start the app
document.addEventListener('DOMContentLoaded', () => {
  window.app = new ExploreSatApp();
});
