package com.example.stage.service;

import com.example.stage.Model.Pointage;
import com.example.stage.Repository.PointageRepository;
import com.example.stage.service.PointageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class PointageServiceImpl implements PointageService {

    @Autowired
    private PointageRepository pointageRepository;

    @Override
    public List<Pointage> getAllPointages() {
        return pointageRepository.findAll();
    }

    @Override
    public Pointage getPointageById(String id) {
        Optional<Pointage> pointage = pointageRepository.findById(id);
        return pointage.orElseThrow(() -> new RuntimeException("Pointage avec ID " + id + " introuvable."));
    }
}
