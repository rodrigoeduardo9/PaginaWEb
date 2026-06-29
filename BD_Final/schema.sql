-- =====================================================
-- Script: schema.sql
-- Descripción: Creación de base de datos para sistema de
--              registro de usuarios y seguimiento de salud
-- Motor: MySQL 8.0+
-- =====================================================

CREATE DATABASE IF NOT EXISTS bd_salud
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE bd_salud;

-- =====================================================
-- Tabla: usuarios
-- Almacena la información de registro personal de cada
-- usuario del sistema.
-- =====================================================
CREATE TABLE usuarios (
    id_usuario INT AUTO_INCREMENT,
    nombre_completo VARCHAR(100) NOT NULL,
    correo VARCHAR(100) NOT NULL,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id_usuario),
    CONSTRAINT uq_correo UNIQUE (correo),
    INDEX idx_correo (correo)
) ENGINE=InnoDB;

-- =====================================================
-- Tabla: registros_salud
-- Almacena los datos de los formularios médicos. Un
-- usuario puede tener múltiples registros a lo largo
-- del tiempo.
-- =====================================================
CREATE TABLE registros_salud (
    id_registro INT AUTO_INCREMENT,
    id_usuario INT NOT NULL,
    fecha_formulario TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    peso_kg DECIMAL(5,2) NOT NULL,
    altura_cm DECIMAL(5,2) NOT NULL,
    edad INT NOT NULL,
    nivel_glucosa DECIMAL(5,2) NOT NULL COMMENT 'mg/dL',
    ojos_rojos ENUM('Sí', 'No') NOT NULL,
    consumo_azucares INT NOT NULL,
    consumo_harinas INT NOT NULL,
    imc DECIMAL(5,2) GENERATED ALWAYS AS (peso_kg / POWER((altura_cm / 100), 2)) VIRTUAL,
    PRIMARY KEY (id_registro),
    CONSTRAINT fk_usuario FOREIGN KEY (id_usuario)
        REFERENCES usuarios(id_usuario)
        ON DELETE CASCADE,
    CONSTRAINT chk_consumo_azucares CHECK (consumo_azucares BETWEEN 1 AND 5),
    CONSTRAINT chk_consumo_harinas CHECK (consumo_harinas BETWEEN 1 AND 5),
    INDEX idx_id_usuario (id_usuario)
) ENGINE=InnoDB;
