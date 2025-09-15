package com.example.stage.Repository;



import com.example.stage.Model.Pointage;
import org.springframework.data.mongodb.repository.MongoRepository;

import java.util.List;

public interface PointageRepository extends MongoRepository<Pointage, String> {
    List<Pointage> findUserById(String id);
}