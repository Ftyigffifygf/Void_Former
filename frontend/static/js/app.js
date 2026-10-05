// Unified Field Theory VM - Frontend JavaScript

const API_BASE = '';
const SESSION_ID = 'user-' + Math.random().toString(36).substr(2, 9);

// Global state
let currentProgram = [];
let currentMemory = {};
let charts = {};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initInstructionBuilder();
    initMemoryEditor();
    initControls();
    initCharts();
    loadExamples();
    updateVMStatus();

    // Auto-refresh status every 2 seconds
    setInterval(updateVMStatus, 2000);
});

// Tab Management
function initTabs() {
    // Program tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchTab(tab);
        });
    });

    // Visualization tabs
    document.querySelectorAll('.viz-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const viz = btn.dataset.viz;
            switchViz(viz);
        });
    });
}

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
    document.getElementById(`${tab}-tab`).classList.add('active');
}

function switchViz(viz) {
    document.querySelectorAll('.viz-tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.viz-content').forEach(c => c.classList.remove('active'));

    document.querySelector(`[data-viz="${viz}"]`).classList.add('active');
    document.getElementById(`${viz}-viz`).classList.add('active');
}

// Instruction Builder
function initInstructionBuilder() {
    document.getElementById('add-instruction-btn').addEventListener('click', addInstruction);
}

function addInstruction() {
    const type = document.getElementById('instruction-type').value;
    const operandsStr = document.getElementById('instruction-operands').value;
    const operands = operandsStr ? operandsStr.split(',').map(s => s.trim()) : [];

    currentProgram.push({ op: type, operands });
    renderProgram();
    updateCodeEditor();

    // Clear input
    document.getElementById('instruction-operands').value = '';

    addLogEntry(`Added instruction: ${type}`, 'info');
}

function removeInstruction(index) {
    currentProgram.splice(index, 1);
    renderProgram();
    updateCodeEditor();
}

function renderProgram() {
    const container = document.getElementById('program-instructions');
    container.innerHTML = '';

    currentProgram.forEach((instr, index) => {
        const div = document.createElement('div');
        div.className = 'instruction-item';
        div.innerHTML = `
            <div class="instruction-info">
                <div class="instruction-op">${instr.op}</div>
                <div class="instruction-operands">${instr.operands.join(', ')}</div>
            </div>
            <button class="instruction-remove" onclick="removeInstruction(${index})">✕</button>
        `;
        container.appendChild(div);
    });
}

function updateCodeEditor() {
    const code = JSON.stringify(currentProgram, null, 2);
    document.getElementById('code-editor').value = code;
}

// Memory Editor
function initMemoryEditor() {
    document.getElementById('add-memory-btn').addEventListener('click', addMemory);
}

function addMemory() {
    const key = document.getElementById('mem-key').value;
    const value = parseFloat(document.getElementById('mem-value').value);

    if (key && !isNaN(value)) {
        currentMemory[key] = value;
        renderMemory();

        // Clear inputs
        document.getElementById('mem-key').value = '';
        document.getElementById('mem-value').value = '';

        addLogEntry(`Added memory: ${key} = ${value}`, 'info');
    }
}

function removeMemory(key) {
    delete currentMemory[key];
    renderMemory();
}

function renderMemory() {
    const container = document.getElementById('memory-list');
    container.innerHTML = '';

    Object.entries(currentMemory).forEach(([key, value]) => {
        const div = document.createElement('div');
        div.className = 'memory-item';
        div.innerHTML = `
            <span class="memory-key">${key}</span>
            <span class="memory-value">${value}</span>
            <button class="instruction-remove" onclick="removeMemory('${key}')">✕</button>
        `;
        container.appendChild(div);
    });
}

// Controls
function initControls() {
    document.getElementById('execute-btn').addEventListener('click', executeProgram);
    document.getElementById('reset-btn').addEventListener('click', resetVM);
    document.getElementById('clear-btn').addEventListener('click', clearProgram);
}

async function executeProgram() {
    addLogEntry('Executing program...', 'info');

    try {
        const response = await fetch(`${API_BASE}/api/vm/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: SESSION_ID,
                program: currentProgram,
                memory: currentMemory,
            }),
        });

        const data = await response.json();

        if (data.success) {
            addLogEntry(`Execution complete! ${data.diagnostics.instructions_executed} instructions`, 'success');

            // Update visualizations
            updateFieldVisualization(data.field_data);
            updateEvolutionChart(data.diagnostics);

            // Display measurements
            displayMeasurements(data.final_state.measurements);

            // Update status
            await updateVMStatus();
        } else {
            addLogEntry(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        addLogEntry(`Network error: ${error.message}`, 'error');
    }
}

async function resetVM() {
    try {
        const response = await fetch(`${API_BASE}/api/vm/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: SESSION_ID }),
        });

        const data = await response.json();
        if (data.success) {
            addLogEntry('VM reset successfully', 'success');
            await updateVMStatus();
        }
    } catch (error) {
        addLogEntry(`Reset error: ${error.message}`, 'error');
    }
}

function clearProgram() {
    currentProgram = [];
    currentMemory = {};
    renderProgram();
    renderMemory();
    updateCodeEditor();
    addLogEntry('Program cleared', 'info');
}

// VM Status
async function updateVMStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/vm/status?session_id=${SESSION_ID}`);
        const data = await response.json();

        if (data.success) {
            const status = data.status;

            // Update status display
            const modeBadge = document.getElementById('status-mode');
            modeBadge.textContent = status.mode;
            modeBadge.setAttribute('data-mode', status.mode);

            document.getElementById('status-pc').textContent = status.program_counter;
            document.getElementById('status-time').textContent = status.time_elapsed.toFixed(4);
            document.getElementById('status-energy').textContent = status.field_energy.toFixed(6);
            document.getElementById('status-entropy').textContent = status.field_entropy.toFixed(4);
            document.getElementById('status-measurements').textContent = status.measurements;
            document.getElementById('status-entanglement').textContent = status.entanglement;

            // Update registers
            displayRegisters(status.registers);

            // Update mode amplitudes chart
            updateModesChart(status.mode_amplitudes);
        }
    } catch (error) {
        console.error('Status update error:', error);
    }
}

function displayRegisters(registers) {
    const container = document.getElementById('registers-list');
    container.innerHTML = '';

    Object.entries(registers).forEach(([name, value]) => {
        const div = document.createElement('div');
        div.className = 'register-item';
        const displayValue = Array.isArray(value) ?
            value.slice(0, 3).map(v => v.toFixed(2)).join(', ') + '...' :
            value;
        div.innerHTML = `
            <span class="register-name">${name}:</span>
            <span class="register-value">${displayValue}</span>
        `;
        container.appendChild(div);
    });
}

function displayMeasurements(measurements) {
    const container = document.getElementById('measurements-list');
    container.innerHTML = '';

    measurements.forEach(m => {
        const div = document.createElement('div');
        div.className = 'measurement-item';
        div.textContent = `Qubit ${m.qubit}: ${m.outcome} (p=${m.probability.toFixed(4)})`;
        container.appendChild(div);
    });
}

// Charts
function initCharts() {
    // Field State Chart
    const fieldCtx = document.getElementById('field-chart').getContext('2d');
    charts.field = new Chart(fieldCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Field Amplitude',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#e0e7ff' } }
            },
            scales: {
                y: { ticks: { color: '#e0e7ff' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { ticks: { color: '#e0e7ff' }, grid: { color: 'rgba(255,255,255,0.1)' } }
            }
        }
    });

    // Mode Amplitudes Chart
    const modesCtx = document.getElementById('modes-chart').getContext('2d');
    charts.modes = new Chart(modesCtx, {
        type: 'bar',
        data: {
            labels: ['Classical', 'Quantum', 'Probabilistic', 'Neural', 'Temporal'],
            datasets: [{
                label: 'Mode Amplitude',
                data: [0.2, 0.2, 0.2, 0.2, 0.2],
                backgroundColor: [
                    '#3b82f6',
                    '#8b5cf6',
                    '#ec4899',
                    '#10b981',
                    '#f59e0b'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: { color: '#e0e7ff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    ticks: { color: '#e0e7ff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            }
        }
    });

    // Energy Chart
    const energyCtx = document.getElementById('energy-chart').getContext('2d');
    charts.energy = new Chart(energyCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Energy',
                    data: [],
                    borderColor: '#10b981',
                    yAxisID: 'y',
                },
                {
                    label: 'Entropy',
                    data: [],
                    borderColor: '#f59e0b',
                    yAxisID: 'y1',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#e0e7ff' } }
            },
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    ticks: { color: '#10b981' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    ticks: { color: '#f59e0b' },
                    grid: { drawOnChartArea: false }
                },
                x: {
                    ticks: { color: '#e0e7ff' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            }
        }
    });

    // Evolution Chart
    const evolutionCtx = document.getElementById('evolution-chart').getContext('2d');
    charts.evolution = new Chart(evolutionCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Field Evolution',
                data: [],
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                fill: true,
                tension: 0.4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#e0e7ff' } }
            },
            scales: {
                y: { ticks: { color: '#e0e7ff' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                x: { ticks: { color: '#e0e7ff' }, grid: { color: 'rgba(255,255,255,0.1)' } }
            }
        }
    });
}

function updateFieldVisualization(fieldData) {
    if (!fieldData || !fieldData.real) return;

    charts.field.data.labels = fieldData.real.map((_, i) => i);
    charts.field.data.datasets[0].data = fieldData.real;
    charts.field.update();
}

function updateModesChart(modeAmplitudes) {
    charts.modes.data.datasets[0].data = [
        modeAmplitudes.classical,
        modeAmplitudes.quantum,
        modeAmplitudes.probabilistic,
        modeAmplitudes.neural,
        modeAmplitudes.temporal,
    ];
    charts.modes.update();
}

function updateEvolutionChart(diagnostics) {
    const energyTrace = diagnostics.field_energy_trace;
    const entropyTrace = diagnostics.field_entropy_trace;

    charts.energy.data.labels = energyTrace.map((_, i) => i);
    charts.energy.data.datasets[0].data = energyTrace;
    charts.energy.data.datasets[1].data = entropyTrace;
    charts.energy.update();

    charts.evolution.data.labels = energyTrace.map((_, i) => i);
    charts.evolution.data.datasets[0].data = energyTrace;
    charts.evolution.update();
}

// Examples
async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE}/api/examples`);
        const data = await response.json();

        if (data.success) {
            displayExamples(data.examples);
        }
    } catch (error) {
        console.error('Failed to load examples:', error);
    }
}

function displayExamples(examples) {
    const container = document.getElementById('examples-list');
    container.innerHTML = '';

    examples.forEach(example => {
        const div = document.createElement('div');
        div.className = 'example-item';
        div.innerHTML = `
            <div class="example-name">${example.name}</div>
            <div class="example-description">${example.description}</div>
        `;
        div.addEventListener('click', () => loadExample(example));
        container.appendChild(div);
    });
}

function loadExample(example) {
    currentProgram = example.program;
    currentMemory = example.memory || {};

    renderProgram();
    renderMemory();
    updateCodeEditor();

    // Switch to visual tab
    switchTab('visual');

    addLogEntry(`Loaded example: ${example.name}`, 'success');
}

// Logging
function addLogEntry(message, type = 'info') {
    const log = document.getElementById('execution-log');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;

    // Keep only last 50 entries
    while (log.children.length > 50) {
        log.removeChild(log.firstChild);
    }
}

// Make functions global for onclick handlers
window.removeInstruction = removeInstruction;
window.removeMemory = removeMemory;
