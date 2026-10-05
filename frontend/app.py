"""Unified Field Theory Virtual Machine - Web Frontend with Post-Quantum Security."""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import torch
import sys
import os
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_field import (
    create_universal_vm,
    Instruction,
    InstructionType,
    FieldMode,
    UnifiedOperator,
)
from voidformer.security.pqc import KyberKEM

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

kyber = KyberKEM()
vm_instances = {}


def get_vm(session_id='default'):
    if session_id not in vm_instances:
        vm_instances[session_id] = create_universal_vm(
            d_classical=64,
            d_quantum=16,
            d_probabilistic=32,
            d_neural=128,
        )
    return vm_instances[session_id]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/pqc/key_exchange', methods=['POST'])
def pqc_key_exchange():
    data = request.json or {}
    client_pk_b64 = data.get("client_public_key_b64", "")
    try:
        client_pk = base64.b64decode(client_pk_b64)
        ciphertext, shared_secret = kyber.encapsulate(client_pk)
        return jsonify({
            'success': True,
            'ciphertext_b64': base64.b64encode(ciphertext).decode('utf-8'),
            'shared_secret_hash_b64': base64.b64encode(shared_secret[:16]).decode('utf-8'),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/vm/create', methods=['POST'])
def create_vm():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    config = data.get('config', {})
    d_classical = config.get('d_classical', 64)
    d_quantum = config.get('d_quantum', 16)
    d_probabilistic = config.get('d_probabilistic', 32)
    d_neural = config.get('d_neural', 128)

    vm_instances[session_id] = create_universal_vm(
        d_classical=d_classical,
        d_quantum=d_quantum,
        d_probabilistic=d_probabilistic,
        d_neural=d_neural,
    )
    return jsonify({
        'success': True,
        'session_id': session_id,
        'config': {
            'd_classical': d_classical,
            'd_quantum': d_quantum,
            'd_probabilistic': d_probabilistic,
            'd_neural': d_neural,
            'd_unified': d_classical + d_quantum + d_probabilistic + d_neural,
        }
    })


@app.route('/api/vm/status', methods=['GET'])
def vm_status():
    session_id = request.args.get('session_id', 'default')
    vm = get_vm(session_id)
    field_state = vm.field.field_state.detach()
    energy = vm.field.compute_field_energy().item()
    entropy = vm.field.compute_field_entropy().item()
    mode_amplitudes = vm.field.mode_amplitudes_normalized().tolist()

    return jsonify({
        'success': True,
        'status': {
            'mode': vm.state.mode.value,
            'program_counter': vm.state.program_counter,
            'time_elapsed': vm.state.time_elapsed,
            'field_energy': energy,
            'field_entropy': entropy,
            'mode_amplitudes': {
                'classical': mode_amplitudes[0],
                'quantum': mode_amplitudes[1],
                'probabilistic': mode_amplitudes[2],
                'neural': mode_amplitudes[3],
                'temporal': mode_amplitudes[4],
            },
            'registers': {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in vm.state.registers.items()},
            'memory_keys': list(vm.state.memory.keys()),
            'measurements': len(vm.state.measurement_history),
            'entanglement': len(vm.state.entanglement_map),
        }
    })


@app.route('/api/vm/execute', methods=['POST'])
def execute_program():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    program_json = data.get('program', [])
    vm = get_vm(session_id)

    program = []
    for instr_data in program_json:
        op = InstructionType(instr_data['op'])
        operands = instr_data.get('operands', [])
        program.append(Instruction(op, operands))

    try:
        final_state, diagnostics = vm.execute_program(program)
        field_state = vm.field.field_state.detach()
        field_data = {
            'real': field_state.real.flatten()[:100].tolist(),
            'imag': field_state.imag.flatten()[:100].tolist() if torch.is_complex(field_state) else None,
        }
        return jsonify({
            'success': True,
            'final_state': {
                'mode': final_state.mode.value,
                'time_elapsed': final_state.time_elapsed,
                'registers': {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in final_state.registers.items()},
                'measurements': final_state.measurement_history,
            },
            'diagnostics': {
                'instructions_executed': diagnostics['instructions_executed'],
                'mode_transitions': diagnostics['mode_transitions'],
                'measurements_performed': diagnostics['measurements_performed'],
                'field_energy_trace': diagnostics['field_energy_trace'][:100],
                'field_entropy_trace': diagnostics['field_entropy_trace'][:100],
            },
            'field_data': field_data,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/vm/field', methods=['GET'])
def get_field_state():
    session_id = request.args.get('session_id', 'default')
    vm = get_vm(session_id)
    field_state = vm.field.field_state.detach()

    classical_part = vm.field.project_to_mode(FieldMode.CLASSICAL)
    quantum_part = vm.field.project_to_mode(FieldMode.QUANTUM)
    prob_part = vm.field.project_to_mode(FieldMode.PROBABILISTIC)
    neural_part = vm.field.project_to_mode(FieldMode.NEURAL)

    def tensor_to_list(t, max_len=50):
        flat = t.real.flatten() if torch.is_complex(t) else t.flatten()
        return flat[:max_len].tolist()

    return jsonify({
        'success': True,
        'field': {
            'full': tensor_to_list(field_state, 200),
            'classical': tensor_to_list(classical_part),
            'quantum': tensor_to_list(quantum_part),
            'probabilistic': tensor_to_list(prob_part),
            'neural': tensor_to_list(neural_part, 100),
        },
        'properties': {
            'energy': vm.field.compute_field_energy().item(),
            'entropy': vm.field.compute_field_entropy().item(),
            'norm': field_state.norm().item(),
        }
    })


@app.route('/api/examples', methods=['GET'])
def get_examples():
    examples = [
        {
            'id': 'classical_arithmetic',
            'name': 'Classical Arithmetic',
            'description': 'Compute (5 + 3) * 2 using classical instructions',
            'program': [
                {'op': 'set_mode', 'operands': ['classical']},
                {'op': 'load', 'operands': ['a', 'mem_a']},
                {'op': 'load', 'operands': ['b', 'mem_b']},
                {'op': 'add', 'operands': ['temp', 'a', 'b']},
                {'op': 'load', 'operands': ['c', 'mem_c']},
                {'op': 'mul', 'operands': ['result', 'temp', 'c']},
            ],
            'memory': {'mem_a': 5.0, 'mem_b': 3.0, 'mem_c': 2.0},
        },
        {
            'id': 'quantum_superposition',
            'name': 'Quantum Superposition',
            'description': 'Create Bell state with H and CNOT gates',
            'program': [
                {'op': 'set_mode', 'operands': ['quantum']},
                {'op': 'hadamard', 'operands': [0]},
                {'op': 'cnot', 'operands': [0, 1]},
                {'op': 'measure', 'operands': [0]},
                {'op': 'measure', 'operands': [1]},
            ],
            'memory': {},
        },
    ]
    return jsonify({'success': True, 'examples': examples})


@app.route('/api/operators/apply', methods=['POST'])
def apply_operator():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    operator_type = data.get('operator_type')
    vm = get_vm(session_id)

    try:
        if operator_type == 'unified':
            op = UnifiedOperator(
                d_classical=vm.field.d_classical,
                d_quantum=vm.field.d_quantum,
                d_probabilistic=vm.field.d_probabilistic,
                d_neural=vm.field.d_neural,
            )
            with torch.no_grad():
                result = op(vm.field.field_state)
            vm.field.field_state = result
            return jsonify({'success': True, 'message': 'Unified operator applied', 'norm': result.norm().item()})
        else:
            return jsonify({'success': False, 'error': 'Unknown operator type'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/vm/reset', methods=['POST'])
def reset_vm():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    if session_id in vm_instances:
        del vm_instances[session_id]
    get_vm(session_id)
    return jsonify({'success': True, 'message': 'VM reset successfully'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
