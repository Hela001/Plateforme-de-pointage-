package com.example.stage.service;

import com.example.stage.Model.Pointage;
import java.util.List;

public interface PointageService {
    List<Pointage> getAllPointages();
    Pointage getPointageById(String id);
}
