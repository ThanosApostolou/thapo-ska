<script setup lang="ts">
import { GlobalState } from '@/domain/global_state/global_state';

const globalState = GlobalState.instance();


</script>

<template>
  <footer class="footer p-2 bg-neutral text-neutral-content">
    profile: {{ globalState.envConfig.envProfile }}
  </footer>
</template>