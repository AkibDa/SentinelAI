import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Sphere, MeshDistortMaterial, Float } from '@react-three/drei';
import * as THREE from 'three';
import { useStore } from '../store/useStore';

function AnimatedSpheres() {
  const result = useStore((state) => state.result);
  const color = result?.verdict === 'Fake' ? '#ff4444' : '#10b981';
  
  return (
    <group>
      <Float speed={1.5} rotationIntensity={1} floatIntensity={2}>
        <Sphere args={[1, 64, 64]} position={[-2, 1, -2]}>
          <MeshDistortMaterial
            color={color}
            speed={2}
            distort={0.4}
            radius={1}
            opacity={0.15}
            transparent
          />
        </Sphere>
      </Float>
      
      <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
        <Sphere args={[1.5, 64, 64]} position={[2, -1, -3]}>
          <MeshDistortMaterial
            color={color}
            speed={1.5}
            distort={0.3}
            radius={1}
            opacity={0.1}
            transparent
          />
        </Sphere>
      </Float>

      <Float speed={1} rotationIntensity={2} floatIntensity={0.5}>
        <Sphere args={[0.8, 64, 64]} position={[0, 2, -4]}>
          <MeshDistortMaterial
            color={color}
            speed={3}
            distort={0.5}
            radius={1}
            opacity={0.12}
            transparent
          />
        </Sphere>
      </Float>
    </group>
  );
}

function Grid() {
  const gridRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (gridRef.current) {
      gridRef.current.rotation.x = -Math.PI / 2 + Math.sin(state.clock.elapsedTime * 0.1) * 0.1;
      gridRef.current.position.y = -2 + Math.cos(state.clock.elapsedTime * 0.2) * 0.1;
    }
  });

  return (
    <group ref={gridRef}>
      <gridHelper args={[100, 50, '#333', '#111']} />
    </group>
  );
}

export default function ThreeBackground() {
  return (
    <div className="fixed inset-0 -z-10 bg-black">
      <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <AnimatedSpheres />
        <Grid />
        <fog attach="fog" args={['#000', 5, 15]} />
      </Canvas>
    </div>
  );
}
