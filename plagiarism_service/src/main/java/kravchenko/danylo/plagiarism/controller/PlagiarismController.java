package kravchenko.danylo.plagiarism.controller;

import kravchenko.danylo.plagiarism.dto.PlagiarismRequestItem;
import kravchenko.danylo.plagiarism.dto.PlagiarismResponseItem;
import kravchenko.danylo.plagiarism.service.PlagiarismService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
public class PlagiarismController {

    private final PlagiarismService plagiarismService;

    @ResponseBody
    @PostMapping("/plagiarism")
    public List<PlagiarismResponseItem> plagiarism(@RequestBody final PlagiarismRequestItem item) {
        return plagiarismService.analyzePlagiarism(item);
    }
}
