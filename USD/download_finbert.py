#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FinBERT 모델 다운로드 스크립트
"""

import os
import sys

# SSL 검증 비활성화
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

def download_finbert(model_name='yiyanghkust/finbert-tone', save_dir='./finbert'):
    """FinBERT 모델을 다운로드하고 로컬에 저장"""
    try:
        print(f"모델 다운로드 시작: {model_name}")
        print("이 작업은 몇 분 정도 소요될 수 있습니다...")
        
        # 토크나이저 다운로드
        print("\n1. 토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✓ 토크나이저 다운로드 완료")
        
        # 모델 다운로드
        print("\n2. 모델 다운로드 중...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("   ✓ 모델 다운로드 완료")
        
        # 로컬에 저장
        print(f"\n3. 로컬에 저장 중: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        print("   ✓ 저장 완료")
        
        # 테스트
        print("\n4. 모델 테스트 중...")
        test_text = "stocks rallied and the company reported strong earnings"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=1)
        print(f"   테스트 입력: {test_text}")
        print(f"   예측 결과: {predictions.item()}")
        print("   ✓ 모델이 정상 작동합니다")
        
        print(f"\n✓✓✓ 모든 작업 완료! ✓✓✓")
        print(f"모델 위치: {os.path.abspath(save_dir)}")
        return True
        
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # 여러 모델 옵션 시도
    models_to_try = [
        'yiyanghkust/finbert-tone',
        'ProsusAI/finbert'
    ]
    
    for model_name in models_to_try:
        print(f"\n{'='*60}")
        print(f"시도 중: {model_name}")
        print('='*60)
        
        success = download_finbert(model_name, save_dir='./finbert')
        if success:
            print(f"\n성공적으로 {model_name}을 다운로드했습니다!")
            break
        else:
            print(f"\n{model_name} 다운로드 실패. 다음 모델 시도...")
    else:
        print("\n모든 모델 다운로드 실패")
        sys.exit(1)
